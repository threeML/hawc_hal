#!/usr/bin/env bash

if [ ! -d "$HOME/miniconda/envs/test-environment" ]; then

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

    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda create -q -n test-environment -c conda-forge -c threeml python=$TRAVIS_PYTHON_VERSION root5 numba numpy scipy astropy healpy astromodels threeml

    set +x

else

    # Using cache

    echo "##################################"
    echo "INSTALLATION CACHE FOUND. REUSING"
    echo "##################################"

fi