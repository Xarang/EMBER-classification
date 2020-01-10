#include "kmeans.h"

#include <immintrin.h>
#include <omp.h>
#include <time.h>

#pragma GCC target("avx")

#define RANDOM_SEED 512

double distance_computation_occurences = 0;

/*
** euclidian distance between 2 vectors of dimension dim
*/
inline double distance(float *vec1, float *vec2, unsigned dim) 
{
    distance_computation_occurences++;
    double dist = 0;
    unsigned vector_size = 8;
    __m256i index = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (unsigned i = 0; i < dim / vector_size; i++)
    {
        //vectorization of double d = vec[i] - vec2[i]
        unsigned base_index = i * vector_size;
        __m256 arr1 = _mm256_i32gather_ps(vec1 + base_index, index, sizeof(float));
        __m256 arr2 = _mm256_i32gather_ps(vec2 + base_index, index, sizeof(float));
        
        __m256 sub_arr = _mm256_sub_ps(arr1, arr2);
        __m256 mul_arr = _mm256_mul_ps(sub_arr, sub_arr);
        double sum = mul_arr[0] + mul_arr[1]
            +   mul_arr[2] + mul_arr[3]
            +   mul_arr[4] + mul_arr[5]
            +   mul_arr[6] + mul_arr[7];
        dist += sum;
    }
    return sqrt(dist);
}

/*
** return closest (min(distance)) cluster of vector `vec` of dimensions `dim`
** computes distance between vector vec and mean vector of each cluster
** sets the distance between the vector and its closest cluster in error `e`
** (`ideally` each vectors should be exactly alligned on their cluster,
** any difference is computed as error)
*/
inline unsigned char classify(float *vec, double *error, struct kmeans_params *p) 
{
    unsigned char min = 0;
    float dist, dist_min = FLT_MAX;

    //computes distance to each centroid
    for (unsigned i = 0; i < p->k; ++i) 
    {
        dist = distance(vec, p->means + i * p->vec_dim, p->vec_dim);
        if (dist < dist_min) 
        {
            dist_min = dist;
            min = i;
        }
    }

    *error = dist_min;
    return min;
}

static inline void print_result(int iter, double time, float err)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"error\": \"%f\"}\n", iter, time, err);
    else
        printf("Iteration: %d, Time: %lf, Error: %f\n", iter, time, err);
}

static inline void compute_means_card(struct kmeans_params *p)
{
    //initialise all mean vectors using our initialisation vectors
    memcpy(p->means, p->means_init, sizeof(float) * p->vec_dim * p->k);
    //obtain the mean vector by diving all its singular values by the amount of vectors in the cluster
    for (unsigned i = 0; i < p->k; ++i)
    {
        //#pragma omp parallel for
        for (unsigned j = 0; j < p->vec_dim; ++j)
            p->means[i * p->vec_dim + j] /= p->card[i];
    }
}

struct kmeans_params *kmeans_params_init(float *data, unsigned vec_dim, unsigned nb_vec, unsigned k)
{
    double t_init = omp_get_wtime();

    struct kmeans_params *params = malloc(sizeof(struct kmeans_params));
    params->data = data;
    params->vec_dim = vec_dim;
    params->nb_vec = nb_vec;
    params->k = k;
    params->means = calloc(sizeof(float), vec_dim * k);
    params->means_init = calloc(sizeof(float), vec_dim * k);
    params->card = calloc(sizeof(unsigned), k);
    params->error = calloc(sizeof(double), nb_vec);
    params->mark = calloc(sizeof(unsigned char), nb_vec);
    params->c = calloc(sizeof(char), nb_vec);

    //mark vectors AGGRESSIVELY
    params->min_error_to_mark = 700000; // min error to mark a vector as rightly placed
    params->min_error_improvement_to_continue = 50000; //min mean error improvement to continue looping

    #pragma omp parallel for
    for (unsigned i = 0; i < nb_vec; ++i)
    {
        // we use range [0, 1, 2] to facilitate indexing
        params->c[i] = 0;
        params->error[i] = 1;
        params->mark[i] = 1;
    }
    unsigned *values = cluster_initial_2_centroids(params);

    // set initial cluster values
    for (unsigned i = 0; i < k; i++)
    {
        for (unsigned j = 0; j < vec_dim; j++)
        {
            //set the initial
            params->means_init[i * vec_dim + j] = data[values[i] * vec_dim + j];
            params->means[i * vec_dim + j] = data[values[i] * vec_dim + j];
        }
        params->card[i] = 1;
    }
    printf("[KMEANS] structure initialisation done in %f sec\n", omp_get_wtime() - t_init);
    return params;
}

void kmeans_params_free(struct kmeans_params *p)
{
    free(p->means);
    free(p->means_init);
    free(p->card);
    free(p->error);
    free(p->mark);
    free(p);
}


/*
** data: the floats contained in the file
** nb_vec: the amount of vectors to be found in the file
** dim: size of a vector
** k: amount of clusters
** min_err: tolerated error step
** max_iter: tolerated iteration step
*/
unsigned char *kmeans(float *data, unsigned nb_vec, unsigned dim,
                      unsigned char k, unsigned max_iter)
{
    double t_start = omp_get_wtime();

    unsigned iter = 0;
    double previous_iteration_error = DBL_MAX;
    double error_improvement = DBL_MAX;

    struct kmeans_params *p = kmeans_params_init(data, dim, nb_vec, k);
    //as long as we dont reach the maximum iteration number, or the improvement is deemed not enough to justify another iteration
    do
    {
        double t1 = omp_get_wtime();
        // Classify data
        double iteration_total_error = 0.;
        #pragma omp parallel for reduction(+: iteration_total_error)
        for (unsigned i = 0; i < nb_vec; ++i) 
        {
            if (p->mark[i])
            {
                //assign a (new) cluster to all vectors
                double vector_error;
                p->c[i] = classify(data + i * p->vec_dim, &vector_error, p);
                if (vector_error < p->min_error_to_mark)
                {
                    p->mark[i] = 0;
                    p->card[p->c[i]] += 1;
                    for (size_t j = 0; j < p->vec_dim; j++)
                    {
                        //The vector is now considered part of this cluster. we add its value to the mean vector, mean to be computed on
                        p->means_init[p->c[i] * p->vec_dim + j] += data[i * p->vec_dim + j];
                    }
                }
                p->error[i] = vector_error;
            }
            //sum up the errors
            iteration_total_error += p->error[i];
        }
        //printf("Iteration: %d. done classification in %f\n", iter, omp_get_wtime() - t1);
        //double t_classification = omp_get_wtime();

        //update means
        compute_means_card(p);

        //double t_mean_card_computation = omp_get_wtime();
        //printf("Iteration: %d. done mean card computation in %f\n", iter, omp_get_wtime() - t_classification);

        //obtain the mean error
        double iteration_mean_error = iteration_total_error / p->nb_vec;
        double t2 = omp_get_wtime();
        error_improvement = fabs(previous_iteration_error - iteration_mean_error);
        previous_iteration_error = iteration_mean_error;

        print_result(iter, t2 - t1, error_improvement);
        iter++;
    }
    while (iter < max_iter && error_improvement > p->min_error_improvement_to_continue);
    unsigned char *result = p->c;
    kmeans_params_free(p);
    printf("[KMEANS] completed in %f sec\n", omp_get_wtime() - t_start);
    return result;
}

int main(int ac, char *av[])
{
    //srand(RANDOM_SEED);
    srand(time(NULL));

    if (ac != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);
    unsigned k = atoi(av[1]);
    unsigned max_iter = atoi(av[2]);
    //double min_err = atof(av[3]);
    unsigned dim = atoi(av[4]);
    unsigned nb_vec = atoi(av[5]);
    char *datafilename = av[6];
    char *outputfile = av[7];

    printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[6], k, dim, nb_vec);

    float *tab = load_data(datafilename, nb_vec, dim);
    unsigned char * classif = kmeans(tab, nb_vec, dim, k, max_iter);

    write_class_in_float_format(classif, nb_vec, outputfile);

    munmap(tab, nb_vec * dim * sizeof(float));
    free(classif);

    return 0;
}
