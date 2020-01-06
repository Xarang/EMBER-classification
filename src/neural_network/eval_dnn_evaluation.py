import neural_network
import sys
import os

model_filename=os.getenv("DNN_MODEL")
xvalidationfile=os.getenv("VALIDATION_DATA")
yvalidationfile=os.getenv("VALIDATION_LABELS")

if model_filename == None or xvalidationfile == None or yvalidationfile == None:
    print("[EVAL] error: one of 'DNN_MODEL', 'VALIDATION_DATA', 'VALIDATION_LABELS' env variable is not set.")
    print("[EVAL] make sure you run the ./setup script at the root of the project or set these variables manually")
    exit

neural_network.load_and_evaluate(model_filename, xvalidationfile, yvalidationfile, 'dnn')