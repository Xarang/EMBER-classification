#include "kmeans.h"

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

    double counts[2] = { 0 };
    for (unsigned i = 0; i < nb_elt; ++i) 
    {
        // from [0, 1, 2] to [-1, 0, 1]
        counts[data[i]]++;
        float f = data[i]/* - 1*/;
        fwrite(&f, sizeof(float), 1, fp);
    }
    printf("outputted %f vectors in cluster #0\n", counts[0]);
    printf("outputted %f vectors in cluster #1\n", counts[1]);

    fclose(fp);
}
