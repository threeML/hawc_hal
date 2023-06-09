import os
from hawc_hal import HAL, HealpixConeROI
from threeML import *
import astromodels
import astropy.units as u
import pytest
from hawc_hal import HealpixConeROI
from hawc_hal.maptree.map_tree import map_tree_factory
from hawc_hal.region_of_interest import (
    HealpixConeROI, 
    HealpixMapROI
)
from hawc_hal.dec_band_select import dec_index_search
test_data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
response=os.path.join(test_data_path, "geminga_response.root")
ra_source, dec_source = 101.7, 16
data_r = 3
roi = HealpixConeROI(data_radius=data_r,
                             model_radius=data_r + 10.0,
                             ra=ra_source, dec=dec_source)

dec_arange=[roi.ra_dec_center[1]-roi.model_radius.to(u.deg).value, roi.ra_dec_center[1]+roi.model_radius.to(u.deg).value]
response_file=response
#Use the declination band selection module
dec_list = dec_index_search(response_file, dec_arange, use_module=True)
print("Using declination list=", dec_list)

#Using all declination bands 
dec_list = dec_index_search(response_file, dec_arange, use_module=False)
print("Using declination list=", dec_list)