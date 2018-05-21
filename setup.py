from setuptools import setup

setup(

    name='hawc_hal',

    version='0.1',

    packages=['hawc_hal',
              'hawc_hal/healpix_handling',
              'hawc_hal/interpolation'],

    url='https://github.com/giacomov/hawc_hal',

    license='BSD-3.0',

    author='Giacomo Vianello',

    author_email='giacomov@stanford.edu',

    description='Read and handle HAWC data',

    install_requires=['numpy',
                      'healpy',
                      'threeml',
                      'pandas',
                      'healpy',
                      'root_numpy',
                      'six',
                      'astropy',
                      'scipy',
                      'matplotlib',
                      'numba']
)
