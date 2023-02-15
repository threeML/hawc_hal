from __future__ import print_function
from hawc_hal import HAL, HealpixConeROI

# try:
# import ROOT
# ROOT.PyConfig.IgnoreCommandLineOptions = True
# except:
# pass

from threeML import *
import argparse
from collections import namedtuple
import pytest


#def test_plugin_from_root(geminga_maptree, geminga_response, geminga_roi):

#    hal = HAL("HAL", geminga_maptree, geminga_response, geminga_roi)
#V1: Rishi: Add bin_list

@pytest.fixture
def bin_list():
        return bin_list;

def test_plugin_from_root(geminga_maptree, geminga_response, geminga_roi, bin_list):

    hal = HAL( "HAL", geminga_maptree, geminga_response, geminga_roi, bin_list=None )

def test_plugin_from_hd5(maptree, response, roi):

    roi = HealpixConeROI(ra=82.628, dec=22.640, data_radius=5, model_radius=10)
    hal = HAL("HAL", maptree, response, roi, bin_list=None)
