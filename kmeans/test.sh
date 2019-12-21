#!/bin/sh

CURRENT=$PWD
DIR=$(dirname $0)
cd $DIR

OUTPUT_DIR="output"
mkdir "$OUTPUT_DIR" 2>/dev/null

OUTPUT_SUB_DIR="$OUTPUT_DIR/$(date --rfc-email)"
mkdir "$OUTPUT_SUB_DIR" 2>/dev/null

KMEANS_OUTPUT_FILE="$OUTPUT_SUB_DIR/kmeans.out"
ACCURACY_OUTPUT_FILE="$OUTPUT_SUB_DIR/accuracy.out"

make
./kmeans 3 20 1.0 2351 900000 ../ember/Xtrain.dat out.dat 1>"$KMEANS_OUTPUT_FILE"

echo "[KMEANS] outputted kmeans logs in $KMEANS_OUTPUT_FILE"

python3 ../python/accuracy/classification_eval.py out.dat ../ember/Ytrain.dat 1>"$ACCURACY_OUTPUT_FILE"

echo "[KMEANS] outputted accuracy logs in $ACCURACY_OUTPUT_FILE"

rm out.dat

cd $CURRENT
