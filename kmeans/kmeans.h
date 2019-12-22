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
** map file `filename` into memory
*/
float *load_data(char *filename, unsigned nb_vec, unsigned dim);

/*
** writes `data` buffer into file `filename`
*/
void write_class_in_float_format(unsigned char *data,
        unsigned nb_elt, char *filename);



#endif /* MY_KMEANS_H */