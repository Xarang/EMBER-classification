#include <err.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include <unistd.h>

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


double distance(float *vec1, float *vec2, unsigned dim) 
{
    double dist = 0;
    for(unsigned i = 0; i < dim; ++i, ++vec1, ++vec2) 
    {
        double d = *vec1 - *vec2;
        dist += d * d;
    }

    return sqrt(dist);
}

unsigned char classify(float *vec, float *means, unsigned dim,
                       unsigned char K, double *e) 
{
    unsigned char min = 0;
    float dist, dist_min = FLT_MAX;

    for(unsigned i = 0; i < K; ++i) 
    {
        dist = distance(vec, means + i * dim, dim);
        if(dist < dist_min) 
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
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"error\": \"%f\"}\n", iter, time, err);
    else
        printf("Iteration: %d, Time: %lf, Error: %f\n", iter, time, err);
}

unsigned char *kmeans(float *data, unsigned nb_vec, unsigned dim,
                      unsigned char K, double min_err, unsigned max_iter)
{
    unsigned iter = 0;
    double e = 0.;
    double diff_err = DBL_MAX;
    double err = DBL_MAX;

    float *means = malloc(sizeof(float) * dim * K);
    unsigned *card = malloc(sizeof(unsigned) * K);
    unsigned char *c = malloc(sizeof(unsigned char) * nb_vec);

    // Random init of c
    for(unsigned i = 0; i < nb_vec; ++i)
        c[i] = rand() / (RAND_MAX + 1.) * K;

    for(unsigned i = 0; i < dim * K; ++i)
        means[i] = 0.;

    for(unsigned i = 0; i < K; ++i)
        card[i] = 0.;

    for(unsigned i = 0; i < nb_vec; ++i) 
    {
        for(unsigned j = 0; j < dim; ++j)
            means[c[i] * dim + j] += data[i * dim  + j];
        ++card[c[i]];
    }

    for(unsigned i = 0; i < K; ++i)
        for(unsigned j = 0; j < dim; ++j)
            means[i * dim + j] /= card[i];

    while ((iter < max_iter) && (diff_err > min_err)) 
    {
        double t1 = omp_get_wtime();
        diff_err = err;
        // Classify data
        err = 0.;
        for(unsigned i = 0; i < nb_vec; ++i) 
        {
            c[i] = classify(data + i * dim, means, dim, K, &e);
            err += e;
        }

        // update Mean
        for(unsigned i = 0; i < dim * K; ++i)
            means[i] = 0.;

        for(unsigned i = 0; i < K; ++i)
            card[i] = 0.;

        for(unsigned i = 0; i < nb_vec; ++i) 
        {
            for(unsigned j = 0; j < dim; ++j)
                means[c[i] * dim + j] += data[i * dim  + j];
            ++card[c[i]];
        }
        for(unsigned i = 0; i < K; ++i)
            for(unsigned j = 0; j < dim; ++j)
                means[i * dim + j] /= card[i];

        ++iter;
        err /= nb_vec;
        double t2 = omp_get_wtime();
        diff_err = fabs(diff_err - err);

        print_result(iter, t2 - t1, err);
    }

    free(means);
    free(card);

    return c;
}

int main(int ac, char *av[])
{
    if (ac != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);

    unsigned max_iter = atoi(av[2]);
    double min_err = atof(av[3]);
    unsigned k = atoi(av[1]);
    unsigned dim = atoi(av[4]);
    unsigned nb_vec = atoi(av[5]);

    printf("Start kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[6], k, dim, nb_vec);

    float *tab = load_data(av[6], nb_vec, dim);
    unsigned char * classif = kmeans(tab, nb_vec, dim, k, min_err, max_iter);

    write_class_in_float_format(classif, nb_vec, av[7]);

    munmap(tab, nb_vec * dim * sizeof(float));
    free(classif);

    return 0;
}
