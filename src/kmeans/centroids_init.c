#include "kmeans.h"

#include <assert.h>
#include <immintrin.h>

#define SUBSET_SIZE 1000
#define NB_CANDIDATES 32

unsigned *get_subset_indexes(unsigned nb_vec)
{
    unsigned *subset = malloc(sizeof(unsigned) * SUBSET_SIZE);
    unsigned subset_count = 0;
    unsigned reroll = 0;
    while (subset_count < SUBSET_SIZE)
    {
        unsigned r = rand() % nb_vec;
        for (size_t i = 0; i < subset_count; i++)
        {
            if (subset[i] == r)
            {
                reroll = 1;
                break;
            }
        }
        if (reroll)
        {
            reroll = 0;
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
    unsigned reroll = 0;
    while (candidates_count < NB_CANDIDATES)
    {
        unsigned r = rand() % SUBSET_SIZE;
        for (size_t i = 0; i < candidates_count; i++)
        {
            if (candidates[i] == subset[r])
            {
                reroll = 1;
                break;
            }
        }
        if (reroll)
        {
            reroll = 0;
            continue;
        }
        candidates[candidates_count] = subset[r];
        candidates_count++;
    }
    return candidates;
}

double **compute_distance_matrix(float *data, unsigned *candidates, unsigned vec_dim)
{
    double **matrix = calloc(sizeof(void*), NB_CANDIDATES);
    for (unsigned i = 0; i < NB_CANDIDATES; i++)
    {
        double *row = calloc(sizeof(double), NB_CANDIDATES);
        for (unsigned j = 0; j < NB_CANDIDATES; j++)
        {
            row[j] = distance(data + candidates[i] * vec_dim, data + candidates[j] * vec_dim);
        }
        matrix[i] = row;
    }
    return matrix;
}

double *compute_mean_distance_vector(float *data, unsigned *candidates, unsigned *subset, unsigned vec_dim)
{
    double *vector = calloc(sizeof(double), NB_CANDIDATES);
    for (unsigned i = 0; i < NB_CANDIDATES; i++)
    {
        double dist = 0;
        //compute mean distance to each point in subset
        //todo: try Median instead
        for (unsigned j = 0; j < SUBSET_SIZE; j++)
        {
            dist += distance(data + candidates[i] * vec_dim, data + subset[j] * vec_dim);
        }
        vector[i] = dist / SUBSET_SIZE;
    }
    return vector;
}

/*
** fills pair array with pair of candidates of similar
*/
void get_candidate_pairs(double *mean_distance_vector, size_t *nb_candidate_pairs, unsigned (*pairs)[2])
{
    for (unsigned i = 0; i < NB_CANDIDATES; i++)
    {
        for (unsigned j = 0; j < NB_CANDIDATES; j++)
        {
            if (i != j)
            {
                if ((fabs((mean_distance_vector[i] / mean_distance_vector[j]) - 1)) < 0.2) //if distances are at most 0.8/1.2 of each other
                {
                    pairs[*nb_candidate_pairs][0] = i;
                    pairs[*nb_candidate_pairs][1] = j;
                    *nb_candidate_pairs = *nb_candidate_pairs + 1;
                }
            }
        }
    }
}

/*
** Initialises centroids using simplified Kmeans+ initialisation algorithm
** Returns the indexes of the centroids in our vector array.
*/
unsigned *cluster_initial_2_centroids(struct kmeans_params *p)
{
    printf("[CENTROID INIT] entered init function\n");
    unsigned *subset = get_subset_indexes(p->nb_vec);
    unsigned *candidates = get_candidates(subset);
    unsigned *centroids = calloc(sizeof(unsigned), 2);

    double **distance_matrix = compute_distance_matrix(p->data, candidates, p->vec_dim);
    double *mean_distance_vector = compute_mean_distance_vector(p->data, candidates, subset, p->vec_dim);
    printf("[CENTROID INIT] computed distance matrix and mean distance vector\n");

    size_t nb_candidate_pairs = 0;
    unsigned pairs[NB_CANDIDATES * NB_CANDIDATES][2] = {0}; // max amount of candidates pair possible

    // get all candidates that have similar mean distance to subset, then evaluate their potential
    // the lower the potential, the better
    get_candidate_pairs(mean_distance_vector, &nb_candidate_pairs, pairs);
    printf("[CENTROID INIT] found %zu potential pairs\n", nb_candidate_pairs);
    assert(nb_candidate_pairs != 0);
    double min_potential = DBL_MAX;
    size_t best_pair = 0;
    for (size_t i = 0; i < nb_candidate_pairs; i++)
    {
        // potential of a pair: difference between distance that formed the pair,
        // and distance between both candidates
        unsigned pair[2] = { pairs[i][0], pairs[i][1] };
        double mean_distance_to_other_points = (mean_distance_vector[pair[0]] + mean_distance_vector[pair[1]]) / 2;
        double potential = fabs(distance_matrix[pair[0]][pair[1]] - mean_distance_to_other_points);
        if (min_potential > potential)
        {
            min_potential = potential;
            best_pair = i;
        }
    }
    unsigned pair[2] = { pairs[best_pair][0], pairs[best_pair][1] };
    centroids[0] = candidates[pair[0]];
    centroids[1] = candidates[pair[1]];
    
    //centroids evaluation
    printf("[CENTROID INIT] mean distance from centroids[0] to all other points in subset: %f\n", mean_distance_vector[pair[0]]);
    printf("[CENTROID INIT] mean distance from centroids[1] to all other points in subset: %f\n", mean_distance_vector[pair[1]]);
    printf("[CENTROID INIT] distance between [0] and [1]: %f\n", distance_matrix[pair[0]][pair[1]]);

    return centroids;
}