import sys
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sys
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def log(message):
    print("[EVAL][KMEANS] {}".format(message))

if len(sys.argv) != 3:
    log("Usage: {} [Path to X] [Path to Y]".format(sys.argv[0]))
    exit(1)

xfile = sys.argv[1]
yfile = sys.argv[2]

kmeans_directory = os.getenv("ESLR_PROJECT_MAIN_DIR") + "src/kmeans/"
kmeans_binary = kmeans_directory + "kmeans"
output_file = 'out.dat'

# make our kmeans and run it
try_execute = subprocess.run(['make', '--directory={}'.format(kmeans_directory)])
try_execute = subprocess.run([kmeans_binary, '2', '20', '1.0', '2351', '900000', xfile, output_file])

classif = np.memmap(output_file, dtype=np.float32, mode='c', order ='C')
exact = np.memmap(yfile, dtype=np.float32, mode='r', order ='C')


cards = np.array([ [0.0, 0.0], [0.0, 0.0] ])
for i in range(len(classif)):
    if (exact[i] >= 0): #ignore unlabelled data when evaluating
        cards[int(classif[i])][int(exact[i])] += 1.0

print(cards)

# test 1: compare repartition in both clusters;
# the more balanced it is, higher the score
card_0 = cards[0][0] + cards[0][1]
card_1 = cards[1][0] + cards[1][1]
repartition_ratio = min(card_0, card_1) / max(card_0, card_1)
log("Repartition score Cluster[0] / Cluster[1]: {}".format(repartition_ratio))

# test 2: compare cluster repartitions with 'Ideal' repartions
# 'Ideal' = all data of same labelled clustered together
expected_card_per_label = len(classif) / 2.0
v1 = np.array([ [0.0, expected_card_per_label], [expected_card_per_label, 0.0] ])
v2 = np.array([ [expected_card_per_label, 0.0], [0.0, expected_card_per_label] ])

def score(card):
    # Compute distance to closest ideal vector
    score = min(euclidean(card, v1[0]), euclidean(card, v1[1]),\
                    euclidean(card, v2[0]), euclidean(card, v2[1]))
    # Compare this distance with the worse distance possible
    score /= expected_card_per_label
    return 1 - score

log("Cluster 0 score: {}".format(score(cards[0])))
log("Cluster 1 score: {}".format(score(cards[1])))
