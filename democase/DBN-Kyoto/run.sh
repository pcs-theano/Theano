#!/bin/sh

DATA_SET=dataset1.pkl

if [ ! -f $DATA_SET ]; then
    echo "Can't find dataset for DBN-Kyoto (dataset1.pkl) in current directory"
    echo "Please get the it from dropbox via below link"
    echo "\thttps://www.dropbox.com/s/ocjgzonmxpmerry/dataset1.pkl.7z?dl=0"
    exit
fi

[ ! -d result ] && mkdir result
python DBN_benchmark.py dataset1.pkl 2000 3
