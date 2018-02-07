from setuptools import setup

setup(
    name='hawc_data_reader',
    version='0.1',
    packages=['hawc_data_reader'],
    url='https://github.com/giacomov/hawc_data_reader',
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
                      'matplotlib']
)
