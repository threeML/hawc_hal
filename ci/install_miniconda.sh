#!/usr/bin/env bash

# Make sure we fail in case of errors
set -e

#if [ ! -d "$HOME/miniconda/envs/test-environment" ]; then

# Rebuilding the cache

echo "########################################"
echo "INSTALLATION CACHE NOT FOUND. REBUILDING"
echo "########################################"

set -x

rm -rf $HOME/miniconda

# Install Miniconda appropriate for the system (Mac or Linux)

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

else

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh

fi

# Install miniconda and all the packages

if [[ ${TRAVIS_PYTHON_VERSION} == 2.7 ]]; then
    PKGS="readline root5 root_numpy"
else
    #root not supported yet
    PKGS="root root_numpy"
fi

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda config --add channels conda-forge
conda create -q -n test-environment -c conda-forge -c threeml python=$TRAVIS_PYTHON_VERSION astromodels threeml numba numpy scipy astropy healpy $PKGS

set +x

#else

    # Using cache

#    echo "##################################"
#    echo "INSTALLATION CACHE FOUND. REUSING"
#    echo "##################################"

#fi
