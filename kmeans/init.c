#include "kmeans.h"

#include <immintrin.h>

#define SUBSET_SIZE 1024
#define NB_CANDIDATES 64

unsigned *get_subset_indexes(unsigned nb_vec)
{
    unsigned *subset = malloc(sizeof(unsigned) * SUBSET_SIZE);
    unsigned subset_count = 0;
    while (subset_count < SUBSET_SIZE)
    {
        unsigned r = rand() % nb_vec;
        for (size_t i = 0; i < subset_count; i++)
        {
            if (subset[i] == r)
                continue;
        }
        subset[subset_count] = r;
        subset_count++;
    }
    return subset;
}

unsigned *get_candidates(unsigned *subset)
{
    unsigned *candidates = malloc(sizeof(unsigned) * NB_CANDIDATES);
    unsigned candidates_count = 0;
    while (candidates_count < NB_CANDIDATES)
    {
        unsigned r = rand() % SUBSET_SIZE;
        for (size_t i = 0; i < candidates_count; i++)
        {
            if (candidates[i] == subset[r])
                continue;
        }
        candidates[candidates_count] = subset[r];
        candidates_count++;
    }
    return candidates;
}

/*
** squared euclidian distance between 2 vectors of dimension dim (same as kmeans.c:distance except no sqrt at the end)
*/
static inline double squared_distance(float *vec1, float *vec2, unsigned dim) 
{
    double dist = 0;
    unsigned vector_size = 8;
    __m256i index = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    #pragma omp parallel for reduction (+: dist)
    for (unsigned i = 0; i < dim / vector_size; i++)
    {
        //vectorization of double d = vec[i] - vec2[i]
        unsigned base_index = i * vector_size;
        __m256 arr1 = _mm256_i32gather_ps(vec1 + base_index, index, sizeof(float));
        __m256 arr2 = _mm256_i32gather_ps(vec2 + base_index, index, sizeof(float));
        
        __m256 sub_arr = _mm256_sub_ps(arr1, arr2);
        __m256 mul_arr = _mm256_mul_ps(sub_arr, sub_arr);
        double sum = mul_arr[0] + mul_arr[1]
            +   mul_arr[2] + mul_arr[3]
            +   mul_arr[4] + mul_arr[5]
            +   mul_arr[6] + mul_arr[7];
        dist += sum;
    }
   return dist;
}

/*
** computes squared sum of distances between all points in subset and closest of all centroids
*/
double potential(float *data, unsigned vec_dim, unsigned *subset, unsigned *centroids, unsigned nb_centroids)
{
    double pot = 0;
    for (size_t i = 0; i < SUBSET_SIZE; i++)
    {
        double min = DBL_MAX;
        for (size_t j = 0; j < nb_centroids; j++)
        {
            double dist = squared_distance(data + vec_dim * subset[i], data + vec_dim * centroids[j], vec_dim);
            if (dist < min)
                min = dist;
        }
        pot += min;
    }
    //warnx("exited potential with pot: %f", pot);
    return pot;
}

unsigned *cluster_initial_vectors(struct kmeans_params *p)
{
    unsigned *subset = get_subset_indexes(p->nb_vec);
    unsigned *candidates = get_candidates(subset);
    unsigned *centroids = calloc(sizeof(unsigned), p->k);
    centroids[0] = rand() % p->nb_vec; //TODO: find a better way to find first centroid
    unsigned nb_centroids = 1;
    for (size_t i = 1; i < p->k; i++)
    {
        double min_potential = DBL_MAX;
        unsigned min = 0;
        for (size_t j = 0; j < NB_CANDIDATES; j++)
        {
            //temporarily add candidate to centroids and compute its potential
            centroids[nb_centroids] = candidates[j];
            double pot = potential(p->data, p->vec_dim, subset, centroids, nb_centroids + 1);
            if (pot < min_potential)
            {
                min_potential = pot;
                min = j;
            }
        }
        //candidate 'min' has the 'lowest' potential, hence we pick it
        centroids[nb_centroids] = candidates[min];
        nb_centroids++;
    }
    return centroids;
}