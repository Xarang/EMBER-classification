CURRENT=$PWD
DIR=$(dirname $0)
cd $DIR

make
./kmeans 3 20 1.0 2351 900000 ../ember/Xtrain.dat out.dat

python3 ../python/accuracy/classification_eval.py out.dat ../ember/Ytrain.dat

cd $CURRENT