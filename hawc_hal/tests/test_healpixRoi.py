import pytest
from hawc_hal import HealpixMapROI, HAL
from hawc_hal.maptree import map_tree_factory
from threeML import Model
from astromodels import PointSource, ExtendedSource, Powerlaw, Gaussian_on_sphere
import healpy as hp
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import os

from conftest import check_map_trees


def Sky2Vec( ra, dec ):
  c = SkyCoord( frame = "icrs", ra=ra*u.degree, dec=dec*u.degree )
  theta = (90.0*u.degree-c.dec).to(u.radian).value
  phi = c.ra.to(u.radian).value
  vec = hp.pixelfunc.ang2vec(theta, phi)
  return vec

NSIDE=512

def test_healpixRoi(geminga_maptree,geminga_response):

  #test to make sure writing a model with HealpixMapROI works fine
  ra, dec = 101.7, 16.
  data_radius = 9.
  model_radius = 24.

  m = np.zeros(hp.nside2npix(NSIDE))
  vec = Sky2Vec(ra, dec)
  m[hp.query_disc(NSIDE, vec, (data_radius*u.degree).to(u.radian).value, inclusive=False)] = 1

  #hp.fitsfunc.write_map("roitemp.fits" , m, nest=False, coord="C", partial=False, overwrite=True )  

  map_roi = HealpixMapROI(data_radius=data_radius, ra=ra, dec=dec, model_radius=model_radius, roimap=m)
  #fits_roi = HealpixMapROI(data_radius=data_radius, ra=ra, dec=dec, model_radius=model_radius, roifile="roitemp.fits")
  hawc = HAL("HAWC",geminga_maptree,geminga_response,map_roi)
  hawc.set_active_measurements(1,9)
 
  '''
  Define model: Two sources, 1 point, 1 extended

  Same declination, but offset in RA

  Different spectral idnex, but both power laws
  '''
  pt_shift=3.0
  ext_shift=2.0    
  
  # First soource
  spectrum1 = Powerlaw()
  source1 = PointSource("point", ra=ra+pt_shift,dec=dec,spectral_shape=spectrum1)
  
  spectrum1.K = 1e-12 / (u.TeV * u.cm **2 * u.s)
  spectrum1.piv = 1* u.TeV
  spectrum1.index = -2.3

  spectrum1.piv.fix = True
  spectrum1.K.fix = True
  spectrum1.index.fix = True

  # Second source
  shape = Gaussian_on_sphere(lon0=ra - ext_shift,lat0=dec,sigma=0.3)
  spectrum2 = Powerlaw()
  source2 = ExtendedSource("extended",spatial_shape=shape,spectral_shape=spectrum2)

  spectrum2.K = 1e-12 / (u.TeV * u.cm **2 * u.s)
  spectrum2.piv = 1* u.TeV
  spectrum2.index = -2.0

  spectrum2.piv.fix = True
  spectrum2.K.fix = True
  spectrum2.index.fix = True
  
  shape.lon0.fix = True
  shape.lat0.fix = True
  shape.sigma.fix = True

  model = Model(source1,source2)

  hawc.set_model(model)

  # Write the model map
  model_map_tree=hawc.write_model_map("test.hd5",test_return_map=True)

  # Read the model back 
  hawc_model = map_tree_factory('test.hd5',map_roi)

  # Check written model and read model are the same
  check_map_trees(hawc_model,model_map_tree)


  os.remove( "test.hd5" )
  
  
