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
** structure holding all the informations we need for kmeans computation
** @param c: contains the cluster assignated to each vector in data
** @param means: mean vector of each clusted
** @param means_init: values for each mean vector to be initialised at each iteration
** @param card: vector contained in a cluster
** @param error: array containing last registered error rate for every vector
** @param mark: array containing vectors that should be visited during iterations
** @param min_error_improvement_to_continue: value used to stop iterating when the gained error is deemed not worth the extra computation
** @param min_error_to_mark: distance to centroid we deem low enough to put this vector in our centroid
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

    double min_error_improvement_to_continue;
    double min_error_to_mark;

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

/*
** initialisation method more fitted for k=2
*/
unsigned *cluster_initial_2_centroids(struct kmeans_params *p);


#endif /* MY_KMEANS_H */