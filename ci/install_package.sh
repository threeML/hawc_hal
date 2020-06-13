#!/usr/bin/env bash

# Make sure we fail in case of errors
#set -e

set -x

conda config --set always_yes yes --set changeps1 no
conda info -a
source activate test-environment
hash -r
conda update -c conda-forge -c threeml threeml astromodels
conda install -c conda-forge pytest-cov codecov
#pip install root_numpy
conda install -c conda-forge root_numpy cython
#pip install ChainConsumer
pip install .

set +x