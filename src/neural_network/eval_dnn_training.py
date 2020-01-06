import neural_network
import sys
import os

xtrainfile=os.getenv('TRAINING_DATA')
xvalidationfile=os.getenv('VALIDATION_DATA')
ytrainfile=os.getenv('TRAINING_LABELS')
yvalidationfile=os.getenv('VALIDATION_LABELS')

if xtrainfile == None or xvalidationfile == None or ytrainfile == None or yvalidationfile == None:
    print("[EVAL] error: one of 'TRAINING_DATA', 'VALIDATION_DATA', 'TRAINING_LABELS', 'VALIDATION_LABELS' env variable is not set.")
    print("[EVAL] make sure you run the ./setup script at the root of the project or set these variables manually")
    exit

neural_network.dnn_train_and_save(xtrainfile, ytrainfile, xvalidationfile, yvalidationfile)