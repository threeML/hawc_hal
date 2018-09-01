#!/usr/bin/env bash

set -x

conda config --set always_yes yes --set changeps1 no
conda info -a
source activate test-environment
hash -r
conda update -c conda-forge -c threeml threeml astromodels
conda install -c conda-forge pytest-cov codecov
pip install root_numpy
pip install .

set +x