#include "kmeans.h"

#include <omp.h>
#include <immintrin.h>

#pragma GCC target("avx")

#define TOLERATED_ERROR_THRESHOLD 50.0 //? sort this out



/*
** Initialised means, card and c array
** c: contains the cluster assignated to each vector in data
** means: mean vector of each clusted
** card: vector contained in a cluster
** error: array containing last registered error rate for every vector
** mark: array containing vectors that should be visited during iterations
*/

struct kmeans_params
{
    float *data;
    unsigned char *c;

    //data informations
    unsigned vec_dim;
    unsigned nb_vec;
    unsigned k;

    //computation arrays
    unsigned *card;
    unsigned *card_init;
    float *means;
    float *means_init;
    double *error;

    unsigned char *mark;

    double tolerated_error_threshold;
};


double distance_computation_occurences = 0;

/*
** euclidian distance between 2 vectors of dimension dim
** use case: vec1 -> some vector; vec2 -> mean vector of cluster k
*/
inline double distance(float *vec1, float *vec2, unsigned dim) 
{
    distance_computation_occurences++;
    double dist = 0;
    unsigned vector_size = 8;
    __m256i index = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    #pragma omp parallel for reduction (+: dist)
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
    double sq = sqrt(dist);
    return sq;
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

    //calls distance k times
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

static inline void print_result(int iter, double time, float err, double distance_occurences)
{
        printf("[KMEANS] Iteration: %d, Time: %lf, error_diff: %f, distance computed %f times\n",
                iter, time, err, distance_occurences);
}

static inline void compute_means_card(struct kmeans_params *p)
{
    //initialise all means to 0
    memset(p->means, 0, p->vec_dim * p->k * sizeof(float));
    //initialise all cards to 0
    memset(p->card, 0, p->k * sizeof(unsigned));
    //for each vector in data
    #pragma omp parallel for
    for (unsigned i = 0; i < p->nb_vec; ++i) 
    {
        //for each float in vector
        //#pragma omp parallel for
        for (unsigned j = 0; j < p->vec_dim; ++j)
        {
            //add the values of the vector to the mean of the cluster that contains this vector
            p->means[p->c[i] * p->vec_dim + j] += p->data[i * p->vec_dim  + j];
        }
        p->card[p->c[i]] += 1;
    }

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
    params->card_init = calloc(sizeof(unsigned), k);
    params->error = calloc(sizeof(double), nb_vec);
    params->mark = calloc(sizeof(unsigned char), nb_vec);
    params->c = calloc(sizeof(char), nb_vec);

    params->tolerated_error_threshold = 50.0;

    //#pragma omp parallel for
    for (unsigned i = 0; i < nb_vec; ++i)
    {
        // Random init of c
        // we use range [0, 1, 2] to facilitate indexing
        params->c[i] = (rand() % k);
        //warnx("%d", params->c[i]);
        params->error[i] = DBL_MAX;
        params->mark[i] = 1;
    }
    compute_means_card(params);
    printf("[KMEANS] structure initialisation done in %f sec\n", omp_get_wtime() - t_init);
    return params;
}

void kmeans_params_free(struct kmeans_params *p)
{
    free(p->means);
    free(p->means_init);
    free(p->card);
    free(p->card_init);
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
                      unsigned char k, double min_err, unsigned max_iter)
{
    warnx("[KMEANS] entered program.");
    double t_start = omp_get_wtime();

    unsigned iter = 0;
    double e = 0.;
    double diff_err = DBL_MAX;
    double err = DBL_MAX;


    struct kmeans_params *p = kmeans_params_init(data, dim, nb_vec, k);
    //as long as we dont reach the maximum iteration number, or we are not satistied by our results..
    while ((iter < max_iter) && (diff_err > min_err))
    {
        double occ1 = distance_computation_occurences;
        double t1 = omp_get_wtime();
        diff_err = err;
        // Classify dataTOLERATED_ERROR_THRESHOLD
        err = 0.;
        #pragma omp parallel for reduction(+: err)
        for (unsigned i = 0; i < nb_vec; ++i) 
        {
            if (p->mark[i])
            {
                //assign a (new) cluster to all vectors
                p->c[i] = classify(data + i * dim, &err, p);
                if (p->error[i] - e < p->tolerated_error_threshold)
                {
                    p->mark[i] = 0;
                }
                p->error[i] = e;
            }
            //sum up the errors
            //TODO: better error calculation
            err += p->error[i];
        }
        
        double t_classification = omp_get_wtime();
        printf("Iteration: %d. done classification in %f\n", iter, t_classification - t1);

        //update means
        compute_means_card(p);

        double t_mean_card_computation = omp_get_wtime();
        printf("Iteration: %d. done mean card computation in %f\n", iter, t_mean_card_computation - t_classification);

        ++iter;
        //obtain the mean error
        err /= nb_vec;
        double t2 = omp_get_wtime();
        diff_err = fabs(diff_err - err);

        print_result(iter, t2 - t1, diff_err, distance_computation_occurences - occ1);
    }
    unsigned char *result = p->c;
    kmeans_params_free(p);

    printf("[KMEANS] completed in %f sec\n", omp_get_wtime() - t_start);
    return result;  
}

int main(int ac, char *av[])
{
    if (ac != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);
    unsigned k = atoi(av[1]);
    unsigned max_iter = atoi(av[2]);
    double min_err = atof(av[3]);
    unsigned dim = atoi(av[4]);
    unsigned nb_vec = atoi(av[5]);
    char *datafilename = av[6];
    char *outputfile = av[7];

    printf("[KMEANS] Start kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", datafilename, k, dim, nb_vec);

    float *tab = load_data(datafilename, nb_vec, dim);
    unsigned char * classif = kmeans(tab, nb_vec, dim, k, min_err, max_iter);

    write_class_in_float_format(classif, nb_vec, outputfile);

    munmap(tab, nb_vec * dim * sizeof(float));
    free(classif);

    return 0;
}
