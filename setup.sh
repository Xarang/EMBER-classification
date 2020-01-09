#!/bin/sh

[ $# -eq 0 ] && log 'Usage: ./setup.sh ([path to ember training dataset] [path to ember training labels] | --set-variables)' && exit 1

log()
{
    echo "[SETUP] $1"
}

set_env()
{
    PATH_TO_PTH="$PWD/env/lib/python3.6/site-packages/set_env.pth"
    >"$PATH_TO_PTH"
    echo -n "import os; " >> "$PATH_TO_PTH"
    echo -n "os.environ['ESLR_PROJECT_MAIN_DIR']='$PWD/'; " >> "$PATH_TO_PTH"
    echo -n "os.environ['TRAINING_DATA']='$PWD/dataset/Xtraining.dat'; " >> "$PATH_TO_PTH"
    echo -n "os.environ['VALIDATION_DATA']='$PWD/dataset/Xvalidation.dat'; " >> "$PATH_TO_PTH"
    echo -n "os.environ['TRAINING_LABELS']='$PWD/dataset/Ytraining.dat'; " >> "$PATH_TO_PTH"
    echo -n "os.environ['VALIDATION_LABELS']='$PWD/dataset/Yvalidation.dat'; " >> "$PATH_TO_PTH"
    echo -n "os.environ['DNN_MODEL']='$PWD/src/neural_network/dnn/models/04/04'; " >> "$PATH_TO_PTH"
}

SCRIPT_DIR="$(dirname $0)"
cd "$SCRIPT_DIR"

if [ "$1" = "--set-variables" ]; then
    log 'Skipping dataset creation'
    set_env
    log 'Setted env variables'
    exit 0
fi

# create virtual env
rm -rf env/ 2>/dev/null
python3.6 -m venv python/env
log '(1/4) Created virtual env'

# install requirements
cd python/env
. bin/activate
cd -
pip3 install -r python/requirements.txt
log '(2/4) Installed requirements'

# set environment variables
set_env
log '(3/4) Set environment variables'


# fill environment variables with actual files
PATH_TO_DATASET="$1"
PATH_TO_EXTRACT_DATASET_SCRIPT='utils/extract_datasets.py'
python3 "$PATH_TO_EXTRACT_DATASET_SCRIPT" "$1" "$2"
log '(4/4) Finished dataset extraction'

log 'Setup completed'