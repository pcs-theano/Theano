#!/bin/sh

cur_dir=`pwd`

export PYTHONPATH=${cur_dir}:$PYTHONPATH
cd is13

### specified Threads numbers for better performance
echo "\n#### RNNSLU-Word_Embeddings elman-forward ####"
MKL_NUM_THREADS=6 OMP_NUM_THREADS=6 python examples/elman-forward.py

### specified Threads numbers for better performance
echo "\n#### RNNSLU-Word_Embeddings jordan-forward ####"
MKL_NUM_THREADS=6 OMP_NUM_THREADS=6 python examples/jordan-forward.py
