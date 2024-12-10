#!/usr/bin/env bash

wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d mnist.scale.bz2
bzip2 -d mnist.scale.t.bz2

cat mnist.scale > mnist_780
head -n 2048 mnist.scale.t >> mnist_780