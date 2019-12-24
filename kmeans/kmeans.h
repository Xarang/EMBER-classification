#ifndef MY_KMEANS_H
#define MY_KMEANS_H

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
#include <unistd.h>


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
    double min_error_improvement_ratio_to_mark;
    double min_closest_to_next_ratio_to_mark;
};

/*
** map file `filename` into memory
*/
float *load_data(char *filename, unsigned nb_vec, unsigned dim);

/*
** writes `data` buffer into file `filename`
*/
void write_class_in_float_format(unsigned char *data,
        unsigned nb_elt, char *filename);

/*
** returns indexes of vectors to base initial clusters on utilizing
** cluster potential calculation
*/
unsigned *cluster_initial_vectors(struct kmeans_params *p);


#endif /* MY_KMEANS_H */