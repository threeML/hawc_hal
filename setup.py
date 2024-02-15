import os
import os.path

from setuptools import setup

# Create list of data files


def find_data_files(directory):
    paths = []

    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))

    return paths


extra_files = find_data_files("hawc_hal/tests/data")


setup(
    name="hawc_hal",
    version="1.0",
    packages=[
        "hawc_hal",
        "hawc_hal/convolved_source",
        "hawc_hal/healpix_handling",
        "hawc_hal/interpolation",
        "hawc_hal/response",
        "hawc_hal/maptree",
        "hawc_hal/psf_fast",
        "hawc_hal/region_of_interest",
        "hawc_hal/convenience_functions",
        "hawc_hal/tests",
        "scripts",
    ],
    url="https://github.com/threeML/hawc_hal",
    license="BSD-3.0",
    author="Giacomo Vianello",
    author_email="giacomov@stanford.edu",
    description="Read and handle HAWC data",
    install_requires=[
        "numpy >=1.14",
        "healpy",
        "threeml",
        "astromodels",
        "pandas",
        "healpy",
        "six",
        "astropy",
        "scipy",
        "matplotlib",
        "numba",
        "reproject",
        "tqdm",
        "uproot==5.1.2",
        "awkward",
        "mplhep",
        "hist",
    ],
    entry_points={
        "console_scripts": [
            "hdf5tofits = scripts.hal_hdf5_to_fits:main",
            "halfitpointsrc = scripts.hal_fit_point_source:main",
        ]
    },
    # NOTE: we use '' as package name because the extra_files already contain the full path from here
    package_data={"": extra_files},
)
