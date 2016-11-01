#!/usr/bin/python


import os

def prepare_mkl():
    # for icpc
    result = os.popen('bash get_self_contained_mkl.sh 1').readlines()
    
    # for gnu
    # result = os.popen('bash prepare_mkl.sh 0').readlines()
    
    if len(result) >= 3:
        MKLROOT = result[-3].strip('\n')
        LIBRARY = result[-2].strip('\n')
        OMP = result[-1].strip('\n')

    print MKLROOT
    print LIBRARY
    print OMP

    if len(MKLROOT) != 0 and MKLROOT != os.environ['MKLROOT']:
        os.environ['MKLROOT'] = MKLROOT


if __name__ == '__main__':
    prepare_mkl()
