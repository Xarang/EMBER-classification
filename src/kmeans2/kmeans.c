#include "kmeans.h"

#include <omp.h>
#include <time.h>

#pragma GCC target("avx")

#define RANDOM_SEED 1024

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
        dist = distance(vec, p->means + i * p->vec_dim);
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
    //obtain the mean vector by diving all its singular values by the amount of vectors in the cluster
    //compute distance for all non-marked vectors
    memset(p->means, 0, sizeof(float) * p->vec_dim * p->k);
    memset(p->cards, 0, sizeof(unsigned) * p->k);
    #pragma omp parallel for
    for (size_t i = 0; i < p->nb_vec; i++)
    {
        // add vectors that are not yet marked to their respective cluster
        #pragma omp critical
        p->cards[p->c[i]] += 1;
        // add to the mean vector of cluster assigned to current vector its vector value
        #pragma omp critical
        add_to_vector(p->means + p->c[i] * p->vec_dim, p->data + i * p->vec_dim);
    }
    divide_mean_vectors(p->means, p->cards, p->k, p->vec_dim);
    printf("[CARDS][0] %u\n", p->cards[0]);
    printf("[CARDS][1] %u\n", p->cards[1]);
}

struct kmeans_params *kmeans_params_init(float *data, unsigned vec_dim, unsigned nb_vec, unsigned k, unsigned max_iter)
{
    double t_init = omp_get_wtime();

    struct kmeans_params *params = malloc(sizeof(struct kmeans_params));
    params->data = data;
    params->vec_dim = vec_dim;
    params->nb_vec = nb_vec;
    params->k = k;
    params->means = calloc(sizeof(float), vec_dim * k);
    params->cards = calloc(sizeof(unsigned), k);
    params->c = calloc(sizeof(char), nb_vec);
    params->max_iter = max_iter;
    mask_init(data, vec_dim);
    params->min_error_improvement_to_continue = 0.1;
    params->max_iter = 0;
    unsigned *centroids = cluster_initial_2_centroids(params);
    printf("[KMEANS] got our centroids: %d; %d\n", centroids[0], centroids[1]);

    // set initial cluster values to values of chosen qcentroids
    for (unsigned i = 0; i < k; i++)
    {
        add_to_vector(params->means + i * vec_dim, data + centroids[i] * vec_dim);
    }
    printf("[KMEANS] structure initialisation done in %f sec\n", omp_get_wtime() - t_init);
    free(centroids);
    return params;
}

void kmeans_params_free(struct kmeans_params *p)
{
    free(p->means);
    free(p->cards);
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
    double error_delta = DBL_MAX;
    struct kmeans_params *p = kmeans_params_init(data, dim, nb_vec, k, max_iter);
    //as long as we dont reach the maximum iteration number, or the improvement is deemed not enough to justify another iteration
    do
    {
        double t1 = omp_get_wtime();
        // Classify data
        double iteration_total_error = 0.;
        #pragma omp parallel for reduction(+: iteration_total_error)
        for (unsigned i = 0; i < p->nb_vec; ++i) 
        {
            double vector_error;
            p->c[i] = classify(data + i * p->vec_dim, &vector_error, p);
            iteration_total_error += vector_error;
        }
        //update means
        compute_means_card(p);

        //obtain the mean error
        double iteration_mean_error = iteration_total_error / p->nb_vec;
        error_delta = fabs(previous_iteration_error - iteration_mean_error);
        previous_iteration_error = iteration_mean_error;
        
        double t2 = omp_get_wtime();

        print_result(iter, t2 - t1, error_delta);
        iter++;
    }
    while (iter < p->max_iter && error_delta > p->min_error_improvement_to_continue);
    unsigned char *result = p->c;
    kmeans_params_free(p);

    printf("[KMEANS] completed in %f sec\n", omp_get_wtime() - t_start);
    return result;
}

int main(int ac, char *av[])
{
    srand(RANDOM_SEED);

    if (ac < 8)
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
