#!/usr/bin/env bash

# Make sure we fail in case of errors
#set -e

set -x

conda config --set always_yes yes --set changeps1 no
conda info -a
source activate test-environment
hash -r
#conda update -c conda-forge -c threeml threeml astromodels
conda install -c conda-forge pytest-cov codecov cython
#pip install root_numpy
if [[ ${TRAVIS_PYTHON_VERSION} == 3.7 ]]; then
    pip install "astropy==3.2" "pytest==3.6"
elif [[ "$OSTYPE" == darwin* ]]; then
    conda update -c conda-forge -c threeml threeml astromodels
fi
pip install .

set +x