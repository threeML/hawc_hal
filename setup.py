from setuptools import setup

setup(

    name='hawc_hal',

    version='1.0',

    packages=['hawc_hal',
              'hawc_hal/convolved_source',
              'hawc_hal/healpix_handling',
              'hawc_hal/interpolation',
              'hawc_hal/response',
              'hawc_hal/maptree',
              'hawc_hal/psf_fast',
              'hawc_hal/region_of_interest',
              'hawc_hal/convenience_functions'],

    url='https://github.com/giacomov/hawc_hal',

    license='BSD-3.0',

    author='Giacomo Vianello',

    author_email='giacomov@stanford.edu',

    description='Read and handle HAWC data',

    install_requires=['numpy >=1.14',
                      'healpy',
                      'threeml',
                      'astromodels',
                      'pandas',
                      'healpy',
                      'six',
                      'astropy',
                      'scipy',
                      'matplotlib',
                      'numba',
                      'reproject',
                      'tqdm'
                      ]
)
