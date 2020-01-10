** ESLR PROJECT 2019 **

author: @Xarang

###################################################


Hello, this is my submission for ESLR recruitement session 2019

To setup the project, ** please execute the script 'setup.sh' **
It will create a virtual environment in the python directory and install requirements.
It will also setup a few environment variables in our venv that my evaluation scripts use.
/ ! \ this might take a minute or two as we also create our training set and validation set for further computations.

Once this is done, you can find some premade evaluation scripts in the eval/ directory


####################################################

** evaluation scripts **



# eval_kmeans.py:
    ####################################################
    Important Note: on my setup I had some issues with first execution after compilation taking for some reason a LOT of time on first computations,
    then any further execution of kmeans would take normal amount of time to complete. if you notice this just wait for the end of the computation of
    kmeans, then relaunch the evaluation
    ####################################################
Usage: python3 eval_kmeans.py [path to X] [path to Y] [N = 1]
computes kmeans k=2 on X, then evaluates its cluster repartition with Y:
- equality of vector distribution among clusters
    (each cluster having about the same amount of vectors in them)
- uniformity of each cluster (data of same label clustered together)
Repeats N time and displays mean results

# eval_classifier.py:
Usage: python3 eval_classifier.py
Expected duration: about a minute
runs our classifier. classifier trains on dataset and evaluates on validation set.
a sub process runs in the background to keep track of ressources usage.

# eval_dnn_training.py:
Usage: python3 eval_dnn_training.py
Expected duration: 1hour+
launches our dnn training

# eval_dnn_evalution.py:
Usage: python3 eval_dnn_training.py
loads model from saved model file + weights, then evaluates it on our validation set.