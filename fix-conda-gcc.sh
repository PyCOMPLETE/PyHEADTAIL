#!/bin/bash

# to fix compiler incompatibilities when trying to build under conda - manifests as somethinbg like
# '''
# *.o: file not recognized: file format not recognized
# collect2: error: ld returned 1 exit status
# error: command 'gcc' failed with exit status 1
# '''

conda install gxx_linux-64
conda create -n cc_env gcc_linux-64
conda activate cc_env
