#include "kmeans.h"

#include <omp.h>
#include <immintrin.h>

/*
** Mask contained the indexes of our data values worth computing
** max contained the respective max absolute value for each feature in our 'mask' vector. Used to normalize data
** weights contained multipliers to apply to our normalized data
*/

#define MASK_SIZE 128
size_t mask[MASK_SIZE] = { 0 };//{ 637,621,618,655,654,642,658,665,640,639,632,671,668,633,622,641,643,660,623,497,500,501,508,496,1060,498,975,503,506,510,504,509,505,499,0,507,657,973,511,502,368,1174,954,101,384,748,611,374,400,888,624,1144,114,391,843,390,882,468,951,680,860,375 };
double max[MASK_SIZE] = { 0 };//{ 1,1,1,1,1,1,2,1,3,1,1,1,1,1,1,2,1,2,1,0.118962,0.125614,0.124656,0.118858,0.197613,3,0.145446,2,0.138595,0.128817,0.140979,0.150622,0.135641,0.145124,0.162327,0.999974,0.159584,1,3,0.193959,0.225016,0.736816,3,2,0.144294,0.694824,43.17467,6.584955,0.386184,0.644531,5,1,3,0.139289,0.235887,4,0.401665,7,0.206387,2,255,4,0.323169,0.224523,0.70557 };
double weights[MASK_SIZE] = { 0 };//{ 1.80617997398389,1.50514997831991,1.32905871926422,1.20411998265592,1.10720996964787,1.02802872360024,0.96108193396963,0.903089986991943,0.851937464544562,0.806179973983887,0.764787288825662,0.726998727936262,0.69223662167705,0.660051938305649,0.630088714928206,0.602059991327962,0.575731052605613,0.550907468880581,0.527426373031058,0.505149978319906,0.483960679249968,0.463757293161681,0.444452137966294,0.425968732272281,0.408239965311849,0.391206626013069,0.3748162098249,0.359021942641668,0.343781976084931,0.329058719264224,0.314818280149614,0.301029995663981,0.287666034106,0.274701056941632,0.262111929633611,0.2498774732166,0.237978249916892,0.226396377367077,0.215115366957388,0.204119982655925,0.193396117264151,0.182930683585987,0.1727115184043,0.1627272974977,0.152967460208543,0.143422142302313,0.13408211604817,0.1249387366083,0.115983893955373,0.107209969647868,0.098609797885951,0.090176630349088,0.081904104383098,0.073786214160919,0.065817284489643,0.057991946977686,0.050305118311396,0.04275198042095,0.035327962341743,0.028028723600244,0.02085013897312,0.013788284485633,0.006839424530305, 0.002 };

struct deviation
{
    float val;
    unsigned index;
};

int deviation_compare(void const *a, void const *b)
{
    const struct deviation *dev_a = a;
    const struct deviation *dev_b = b;
    return dev_a->val < dev_b->val;
}

void mask_init(float *data, unsigned nb_vec, unsigned vec_dim)
{
    //get vectors from a subset of our data
    unsigned subset_size = 2000;
    unsigned start_index = rand() % nb_vec - subset_size;

    float **vectors = calloc(sizeof(void*), subset_size);
    //matrix containing values of our selected vectors
    //(subset_size vectors of size vec_dim)

    float *max_absolute_values = calloc(sizeof(float), vec_dim);
    //max absolute value for each feature

    float *means = calloc(sizeof(float), vec_dim);
    //mean value for each feature

    struct deviation *deviations = calloc(sizeof(struct deviation), vec_dim);
    //standard deviation for each feature

    for (unsigned i = 0; i < vec_dim; i++)
    {
        deviations[i].index = i;
    }
    for (unsigned i = 0; i < subset_size; i++)
    {
        float *vector = calloc(sizeof(float), vec_dim);
        for (unsigned j = 0; j < vec_dim; j++)
        {
            float value = data[(start_index + i) * vec_dim + j];
            if (fabs(value) > max_absolute_values[j])
            {
                max_absolute_values[j] = fabs(value);
            }
            vector[j] = value;
            means[j] += value;
        }
        vectors[i] = vector;
    }
    //get means and deviation for each dim
    for (unsigned i = 0; i < vec_dim; i++)
    {
        means[i] /= max_absolute_values[i];
        means[i] /= subset_size;
        float sum = 0;
        for (unsigned j = 0; j < subset_size; j++)
        {
            float val = means[i] - vectors[j][i] / max_absolute_values[i] / subset_size;
            val *= val;
            sum += val;
        }
        deviations[i].val = sqrt(sum / subset_size);
    }
    //sort our deviations in descending orders so that features with the highest deviation come up at the top
    qsort(deviations, vec_dim, sizeof(struct deviation), deviation_compare);
    for (unsigned i = 0; i < 128; i++)
    {
        //printf("DEVIATION[%u]: index: %u; val: %f\n", i, deviations[i].index, deviations[i].val);
    }
    for (unsigned i = 0; i < subset_size; i++)
        free(vectors[i]);
    //select our MASK_SIZE best values
    for (unsigned i = 0; i < MASK_SIZE; i++)
    {
        mask[i] = deviations[i].index;
        max[i] = max_absolute_values[deviations[i].index];
        weights[i] = log(MASK_SIZE) - log(i + 1) + 0.001;
    }
    printf("[MASK] setup our distance mask computation.\n");
    free(vectors);
    free(means);
    free(deviations);
}


void print_arr(__m256 arr)
{
    printf("[ ");
    for(size_t i = 0; i < 8; i++)
    {
        printf("%f,", arr[i]);
    }
    printf(" ]\n");
}

double distance(float *vec1, float *vec2) 
{
    double dist = 0;
    for (unsigned i = 0; i < MASK_SIZE / 8; i++)
    {
        __m256 arr1 = _mm256_set_ps(
            vec1[mask[i * 8 + 0]], vec1[mask[i * 8 + 1]],
            vec1[mask[i * 8 + 2]], vec1[mask[i * 8 + 3]],
            vec1[mask[i * 8 + 4]], vec1[mask[i * 8 + 5]],
            vec1[mask[i * 8 + 6]], vec1[mask[i * 8 + 7]]);
        __m256 arr2 = _mm256_set_ps(
            vec2[mask[i * 8 + 0]], vec2[mask[i * 8 + 1]],
            vec2[mask[i * 8 + 2]], vec2[mask[i * 8 + 3]],
            vec2[mask[i * 8 + 4]], vec2[mask[i * 8 + 5]],
            vec2[mask[i * 8 + 6]], vec2[mask[i * 8 + 7]]);
        __m256 max_mask = _mm256_set_ps(
            max[i * 8 + 0], max[i * 8 + 1],
            max[i * 8 + 2], max[i * 8 + 3],
            max[i * 8 + 4], max[i * 8 + 5],
            max[i * 8 + 6], max[i * 8 + 7]);
        __m256 weights_mask = _mm256_set_ps(
            weights[i * 8 + 0], weights[i * 8 + 1],
            weights[i * 8 + 2], weights[i * 8 + 3],
            weights[i * 8 + 4], weights[i * 8 + 5],
            weights[i * 8 + 6], weights[i * 8 + 7]);
        // normalize data
        arr1 = _mm256_div_ps(arr1, max_mask);
        arr2 = _mm256_div_ps(arr2, max_mask);
        // multiply by weight
        arr1 = _mm256_mul_ps(arr1, weights_mask);
        arr2 = _mm256_mul_ps(arr2, weights_mask);
        // get differences
        __m256 sub_arr = _mm256_sub_ps(arr1, arr2);
        // square the differences
        __m256 mul_arr = _mm256_mul_ps(sub_arr, sub_arr);
        // sum up the differences
        double sum = mul_arr[0] + mul_arr[1]
            +   mul_arr[2] + mul_arr[3]
            +   mul_arr[4] + mul_arr[5]
            +   mul_arr[6] + mul_arr[7];
        dist += sum;
    }
    //printf("distance: %f\n", sqrt(dist));
    return sqrt(dist);
}

/*
** divide mean vector values by their respective cards (to actually obtain the mean)
*/
void divide_mean_vectors(float *means, unsigned *cards, unsigned k, unsigned vec_dim)
{
    for (unsigned i = 0; i < k; ++i)
    {
        for (unsigned j = 0; j < MASK_SIZE; ++j)
            means[i * vec_dim + mask[j]] /= cards[i];
    }
}

void vector_print(float *vec)
{
    printf("VECTOR[");
    for (unsigned i = 0; i < MASK_SIZE; i++)
        printf("%f, ", vec[mask[i]]);
    printf("]\n");
}

/*
** add vector src to vector dest following our mask
*/
void add_to_vector(float *dest, float *src)
{
    for (unsigned i = 0; i < MASK_SIZE; i++)
        dest[mask[i]] += src[mask[i]];
}
