import sys
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sys
import subprocess
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

if len(sys.argv) < 3:
    print("Usage: {} [Path to X] [Path to Y] [nb try = 1]".format(sys.argv[0]))
    exit(1)


xfile = sys.argv[1]
yfile = sys.argv[2]
nb_try = 1
if (len(sys.argv) >= 4):
    nb_try = int(sys.argv[3])

kmeans_directory = os.getenv("ESLR_PROJECT_MAIN_DIR") + "src/kmeans2/"
kmeans_binary = kmeans_directory + "kmeans"
output_file = 'out.dat'

# make our kmeans
try_execute = subprocess.run(['make', '--directory={}'.format(kmeans_directory)])

mean_repartition_score = 0
mean_cluster_0_score = 0
mean_cluster_1_score = 0
mean_computation_time = 0

for i in range(nb_try):
    time_start = time.time()
    try_execute = subprocess.run([kmeans_binary, '2', '20', '1.0', '2351', '900000', xfile, output_file])
    computation_time = time.time() - time_start

    classif = np.memmap(output_file, dtype=np.float32, mode='c', order ='C')
    exact = np.memmap(yfile, dtype=np.float32, mode='r', order ='C')


    cards = np.array([ [0.0, 0.0], [0.0, 0.0] ])
    for j in range(len(classif)):
        if (exact[j] >= 0): #ignore unlabelled data when evaluating
            cards[int(classif[j])][int(exact[j])] += 1.0

    print(cards)
    # test 1: compare repartition in both clusters;
    # the more balanced it is, higher the score
    card_0 = cards[0][0] + cards[0][1]
    card_1 = cards[1][0] + cards[1][1]
    repartition_ratio = min(card_0, card_1) / max(card_0, card_1)

    # test 2: compare cluster repartitions with 'Ideal' repartions
    # 'Ideal' = all data of same labelled clustered together
    v1 = np.array([ [0.0, 1.0], [1.0, 0.0] ])
    v2 = np.array([ [1.0, 0.0], [0.0, 1.0] ])

    def normalize(card):
        return card / max(card[0], card[1])

    def score(card):
        # Compute distance to closest ideal vector
        score = min(euclidean(card, v1[0]), euclidean(card, v1[1]),\
                        euclidean(card, v2[0]), euclidean(card, v2[1]))
        # Compare this distance with the worse distance possible
        return 1 - score
    score_0 = score(normalize(cards[0]))
    score_1 = score(normalize(cards[1]))
    print("[EVAL][KMEANS][{}] computed in {:.2f} seconds ---- cluster repartition : {:.0f}/{:.0f} (score: {:.2f}) ---- cluster[0] score: {:.2f} ---- cluster[1] score: {:.2f}".format(\
        i, computation_time, card_0, card_1, repartition_ratio, score_0, score_1))

    mean_repartition_score += repartition_ratio
    mean_cluster_0_score += score_0
    mean_cluster_1_score += score_1
    mean_computation_time += computation_time


print("[EVAL][KMEANS][RESULTS]")
print("-- mean repartition score: {}".format(mean_repartition_score / nb_try))
print("-- mean cluster 0 score: {}".format(mean_cluster_0_score / nb_try))
print("-- mean cluster 1 score: {}".format(mean_cluster_1_score / nb_try))
print("-- mean computation time: {}".format(mean_computation_time / nb_try))
