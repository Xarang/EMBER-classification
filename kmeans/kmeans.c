#include <err.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include <unistd.h>

#define TOLERATED_ERROR_THRESHOLD 50.0 //? sort this out

/*
** map file `filename` into memory
*/

float *load_data(char *filename, unsigned nb_vec, unsigned dim)
{
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        err(1, "Error while opening %s", filename);

    struct stat st;
    if (fstat(fd, &st) != -1 && nb_vec * dim * sizeof(float) > (size_t) st.st_size)
        errx(1, "Error in parameters");

    void *tab = mmap(NULL, nb_vec * dim * sizeof(float), PROT_READ,
                     MAP_SHARED, fd, 0);
    if (tab == MAP_FAILED)
        err(1, "Error while mmaping `%s`", filename);
    close(fd);

    return tab;
}


/*
** writes `data` buffer into file `filename`
*/
void write_class_in_float_format(unsigned char *data,
        unsigned nb_elt, char *filename) 
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
        err(1, "Cannot create file: `%s`\n", filename);

    for(unsigned i = 0; i < nb_elt; ++i) 
    {
        float f = data[i];
        fwrite(&f, sizeof(float), 1, fp);
    }

    fclose(fp);
}

/*
** euclidian distance between 2 vectors of dimension dim
** use case: vec1 -> some vector; vec2 -> mean vector of cluster k
*/
double distance(float *vec1, float *vec2, unsigned dim) 
{
    double dist = 0;
    #pragma omp parallel for reduction (+:dist)
    for (unsigned i = 0; i < dim; ++i) 
    {
        double d = vec1[i] - vec2[i];
        dist += d * d;
    }

    //warnx("%d: distance: %f", omp_get_num_threads(), dist);
    return sqrt(dist);
}

/*
** return closest (min(distance)) cluster of vector `vec` of dimensions `dim`
** computes distance between vector vec and mean vector of each cluster
** sets the distance between the vector and its closest cluster in error `e`
** (`ideally` each vectors should be exactly alligned on their cluster,
** any difference is computed as error)
*/
unsigned char classify(float *vec, float *means, unsigned dim,
                       unsigned char k, double *e) 
{
    unsigned char min = 0;
    float dist, dist_min = FLT_MAX;

    //calls distance k times
    for (unsigned i = 0; i < k; ++i) 
    {
        dist = distance(vec, means + i * dim, dim);
        if (dist < dist_min) 
        {
            dist_min = dist;
            min = i;
        }
    }

    *e = dist_min;
    return min;
}

static inline void print_result(int iter, double time, float err)
{
        printf("[KMEANS] Iteration: %d, Time: %lf, error_diff: %f\n", iter, time, err);
}

void compute_means_card(float *data, float *means, unsigned *card, unsigned char *c, unsigned nb_vec, unsigned dim, unsigned k)
{
    //initialise all means to 0
    memset(means, 0, dim * k * sizeof(float));
    //initialise all cards to 0
    memset(card, 0, k * sizeof(unsigned));
    //for each vector in data
    #pragma omp parallel for
    for (unsigned i = 0; i < nb_vec; ++i) 
    {
        //for each float in vector
        for (unsigned j = 0; j < dim; ++j)
        {
            //add the values of the vector to the mean of the cluster that contains this vector
            means[c[i] * dim + j] += data[i * dim  + j];
        }
        ++card[c[i]];
    }

    //obtain the mean vector by diving all its singular values by the amount of vectors in the cluster
    for (unsigned i = 0; i < k; ++i)
        for (unsigned j = 0; j < dim; ++j)
            means[i * dim + j] /= card[i];
}

/*
** Initialised means, card and c array
** c: contains the cluster assignated to each vector in data
** means: mean vector of each clusted
** card: vector contained in a cluster
** error: array containing last registered error rate for every vector
** mark: array containing vectors that should be visited during iterations
*/
static void kmeans_init(float *means, unsigned *card, unsigned char *c, double *error, unsigned char *mark,
    float *data, unsigned nb_vec, unsigned dim, unsigned char k)
{
    double t_init = omp_get_wtime();
    #pragma omp parallel for
    for (unsigned i = 0; i < nb_vec; ++i)
    {
        // Random init of c
        c[i] = rand() / (RAND_MAX + 1.) * k;
        error[i] = DBL_MAX;
        mark[i] = 1;
    }
    compute_means_card(data, means, card, c, nb_vec, dim, k);
    printf("[KMEANS] structure initialisation done in %f sec\n", omp_get_wtime() - t_init);
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
    double t_start = omp_get_wtime();

    unsigned iter = 0;
    double e = 0.;
    double diff_err = DBL_MAX;
    double err = DBL_MAX;

    float *means = malloc(sizeof(float) * dim * k);
    unsigned *card = malloc(sizeof(unsigned) * k);
    unsigned char *c = malloc(sizeof(unsigned char) * nb_vec);
    double *error = malloc(sizeof(double) * nb_vec);
    unsigned char *mark = malloc(sizeof(unsigned char) * nb_vec);

    kmeans_init(means, card, c, error, mark, data,nb_vec, dim, k);

    //as long as we dont reach the maximum iteration number, or we are not satistied by our results..
    while ((iter < max_iter) && (diff_err > min_err))
    {
        double t1 = omp_get_wtime();
        diff_err = err;
        // Classify data
        err = 0.;
        #pragma omp parallel for reduction(+: err)
        for (unsigned i = 0; i < nb_vec; ++i) 
        {
            if (mark[i])
            {
                //assign a (new) cluster to all vectors
                c[i] = classify(data + i * dim, means, dim, k, &e);
                if (error[i] - e < TOLERATED_ERROR_THRESHOLD)
                {
                    mark[i] = 0;
                }
                error[i] = e;
            }
            //sum up the errors
            //TODO: better error calculation
            err += error[i];
        }
        
        //update means
        compute_means_card(data, means, card, c, nb_vec, dim, k);

        ++iter;
        //obtain the mean error
        err /= nb_vec;
        double t2 = omp_get_wtime();
        diff_err = fabs(diff_err - err);

        print_result(iter, t2 - t1, diff_err);
    }

    free(means);
    free(card);

    printf("[KMEANS] completed in %f sec\n", omp_get_wtime() - t_start);
    return c;
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

    printf("[KMEANS] Start kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[4], k, dim, nb_vec);

    float *tab = load_data(av[6], nb_vec, dim);
    unsigned char * classif = kmeans(tab, nb_vec, dim, k, min_err, max_iter);

    write_class_in_float_format(classif, nb_vec, av[7]);

    munmap(tab, nb_vec * dim * sizeof(float));
    free(classif);

    return 0;
}
