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
    unsigned *cards;
    unsigned *cards_init;
    float *means;
    float *means_init;
    double *error;

    unsigned char *mark;
    double min_error_improvement_to_continue;
    unsigned max_iter;
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
** euclidean distance using our computation mask
*/
double distance(float *vec1, float *vec2);

/*
** same as distance(), except does not apply sqrt at the end
*/
double squared_distance(float *vec1, float *vec2);

/*
** divides all mean vectors by the card of their respective cluster (to actually get the mean value)
*/
void divide_mean_vectors(float *means, unsigned *cards, unsigned k, unsigned vec_dim);

/*
** dumps vector masked values
*/
void vector_print(float *vec);

/*
** add vectors using our feature mask; useful for mean vector computation
*/
void add_to_vector(float *dest, float *src);

/*
** returns indexes of vectors to base initial clusters on utilizing
** cluster potential calculation
*/
unsigned *cluster_initial_vectors(struct kmeans_params *p);

/*
** returns indexes of vectors to base initial clusters on utilizing
** cluster potential calculation
*/
unsigned *cluster_initial_2_centroids(struct kmeans_params *p);

/*
** when k != 2, use this less effective method
*/
unsigned *cluster_initial_centroids(struct kmeans_params *p);

/*
** computes standard deviations of important values in our data, establishes a better feature vector
**
*/
void mask_init(float *data, unsigned nb_vec, unsigned vec_dim);

#endif /* MY_KMEANS_H */