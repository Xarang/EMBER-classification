#include "kmeans.h"

#include <omp.h>
#include <time.h>

#pragma GCC target("avx")


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
        //printf("[CLASSIFY] distance to %d: %f\n", i, dist);
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
    printf("[KMEANS] Iteration: %d, Time: %lf, error: %f\n",
            iter, time, err);
}

static inline void compute_means_card(struct kmeans_params *p)
{
    //TODO: add some computation back to this function
    //initialise all mean vectors using our initialisation vectors
    //memcpy(p->means, p->means_init, sizeof(float) * p->vec_dim * p->k);
    //memcpy(p->cards, p->cards_init, sizeof(unsigned) * p->k);
    //obtain the mean vector by diving all its singular values by the amount of vectors in the cluster
    //compute distance for all non-marked vectors
    memset(p->means, 0, sizeof(float) * p->vec_dim * p->k);
    memset(p->cards, 0, sizeof(unsigned) * p->k);
    #pragma omp parallel for
    for (size_t i = 0; i < p->nb_vec; i++)
    {
        // add vectors that are not yet marked to their respective cluster
        if (p->mark[i])
        {
            #pragma omp critical
            p->cards[p->c[i]] += 1;
            // add to the mean vector of cluster assigned to current vector its vector value
            #pragma omp critical
            add_to_vector(p->means + p->c[i] * p->vec_dim, p->data + i * p->vec_dim);
        }
    }
    printf("[CARD][0]: %d ; [CARD][1]: %d\n", p->cards[0], p->cards[1]);
    divide_mean_vectors(p->means, p->cards, p->k, p->vec_dim);
    for (size_t i = 0; i < p->k; i++)
    {
        //printf("[CARD %zu] :", i);
        //vector_print(p->means + i * p->vec_dim);
        //printf("\n");
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
    params->cards = calloc(sizeof(unsigned), k);
    params->cards_init = calloc(sizeof(unsigned), k);
    params->error = calloc(sizeof(double), nb_vec);
    params->mark = calloc(sizeof(unsigned char), nb_vec);
    params->c = calloc(sizeof(char), nb_vec);

    //TODO: find out best variables to put there
    params->min_error_to_mark = 0;//TODO: briefly disable vector marking // min error to mark a vector as rightly placed
    params->min_error_ratio_improvement_to_continue = 0.1; //min mean error improvement to continue looping
    params->min_error_improvement_to_mark = 1;
    params->min_error_improvement_to_continue = 0.1;
    //#pragma omp parallel for
    for (unsigned i = 0; i < nb_vec; ++i)
    {
        params->c[i] = 0;
        //warnx("%d", params->c[i]);
        params->error[i] = DBL_MAX;
        params->mark[i] = 1;
    }
    unsigned *values = cluster_initial_2_centroids(params);
    printf("[KMEANS] got our centroids: %d; %d\n", values[0], values[1]);


    // set initial cluster values to values of centroids
    for (unsigned i = 0; i < k; i++)
    {
        //add_to_vector(params->means_init + i * vec_dim, data + values[i] * vec_dim);
        add_to_vector(params->means + i * vec_dim, data + values[i] * vec_dim);
        //params->cards_init[i] = 1;
        //params->cards[i] = 1;
    }
    //initialise mean vectors
    //compute_means_card(params);
    //check centroids infos
    printf("[KMEANS] structure initialisation done in %f sec\n", omp_get_wtime() - t_init);
    return params;
}

void kmeans_params_free(struct kmeans_params *p)
{
    free(p->means);
    free(p->means_init);
    free(p->cards);
    free(p->cards_init);
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
    warnx("[KMEANS] entered program.");
    double t_start = omp_get_wtime();

    unsigned iter = 0;
    double previous_iteration_error = DBL_MAX;
    double error_delta = DBL_MAX;

    struct kmeans_params *p = kmeans_params_init(data, dim, nb_vec, k);
    //as long as we dont reach the maximum iteration number, or the improvement is deemed not enough to justify another iteration
    do
    {
        double mark_occurences = 0;
        double t1 = omp_get_wtime();
        // Classify data
        double iteration_total_error = 0.;
        #pragma omp parallel for reduction(+: iteration_total_error, mark_occurences)
        for (unsigned i = 0; i < nb_vec; ++i) 
        {
            //if (p->mark[i])
            //{
                //assign a (new) cluster to all vectors
                double vector_error;
                p->c[i] = classify(data + i * p->vec_dim, &vector_error, p);
                //dont mark on first computation
                #if 0
                if (vector_error < p->min_error_to_mark)
                {
                    p->mark[i] = 0;
                    //The vector is now considered part of this cluster. we add its value to the mean vector, mean to be computed on
                    #pragma omp critical
                    p->cards_init[p->c[i]] += 1;
                    for (size_t j = 0; j < p->vec_dim; j++)
                    {
                        p->means_init[p->c[i] * p->vec_dim + j] += data[i * p->vec_dim + j];
                    }
                    mark_occurences++;
                }
                p->error[i] = vector_error;
                iteration_total_error += p->error[i];
                #endif
            //}
                iteration_total_error += vector_error;
            //sum up the errors
            //TODO: try to not count marked vectors as error
        }
        double t_classification = omp_get_wtime();
        printf("Iteration: %d. done classification in %f. Marked %f vectors\n", iter, t_classification - t1, mark_occurences);

        //update means
        compute_means_card(p);

        double t_mean_card_computation = omp_get_wtime();
        printf("Iteration: %d. done mean card computation in %f\n", iter, t_mean_card_computation - t_classification);

        //obtain the mean error
        double iteration_mean_error = iteration_total_error / p->nb_vec;
        error_delta = fabs(previous_iteration_error - iteration_mean_error);
        previous_iteration_error = iteration_mean_error;
        
        double t2 = omp_get_wtime();

        print_result(iter, t2 - t1, error_delta);
        iter++;
    }
    while (iter < max_iter && error_delta > p->min_error_improvement_to_continue);
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

    printf("[KMEANS] Start kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", datafilename, k, dim, nb_vec);

    float *tab = load_data(datafilename, nb_vec, dim);
    unsigned char * classif = kmeans(tab, nb_vec, dim, k, max_iter);

    write_class_in_float_format(classif, nb_vec, outputfile);

    munmap(tab, nb_vec * dim * sizeof(float));
    free(classif);

    return 0;
}
