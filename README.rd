multiple approaches to classification of the EMBER dataset: kmeans, sklearn, keras...


###################################################

author: @Xarang

###################################################

To setup the project, ** please execute the script 'setup.sh' **
It will create a virtual environment in the python directory and install requirements.
It will also setup a few environment variables in our venv that my evaluation scripts use.
/ ! \ this might take a minute or two as we also create our training set and validation set for further computations.

Once this is done, you can find some premade evaluation scripts named eval_*

####################################################

** evaluation scripts **

#kmeans
Usage: python3 eval_classification.py [path to .dat] [path to Y]

# kmeans2.py:
Usage: python3 eval_kmeans.py [path to X] [path to Y] [N = 1]

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
