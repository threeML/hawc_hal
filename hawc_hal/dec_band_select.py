from __future__ import division
from builtins import zip
from past.utils import old_div
from builtins import object
import argparse as ap
import numpy as np
import scipy.stats       as stats
import warnings
import os
import sys
import corner as cn
import healpy            as hp
import scipy

from pathlib import Path

import uproot
from threeML.io.logging import setup_logger
from astropy.coordinates import SkyCoord

with warnings.catch_warnings():
    import astromodels
        
log = setup_logger(__name__)
log.propagate = False

def dec_index_search(response_file, dec_var, use_module):
        response = response_file
        with uproot.open(response) as response_file:
            dec_bins_lower_edge = response_file["DecBins/lowerEdge"].array().to_numpy()
            dec_bins_upper_edge = response_file["DecBins/upperEdge"].array().to_numpy()
            dec_bins_center = response_file["DecBins/simdec"].array().to_numpy()
        if use_module==True:
            dec_min = np.amin(dec_var)
            dec_max = np.amax(dec_var)
        else:
            dec_min = -37.5
            dec_max = 77.5

        log.info("Dec min= %.3f" %(dec_min))
        log.info("Dec max= %.3f" %(dec_max))
        #lower_edges = np.array([-37.5, -32.5, -27.5, -22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5], dtype=np.float)
        #upper_edges = np.array([-32.5, -27.5, -22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5], dtype=np.float)
        #centers = np.array([-35., -30., -25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75.], dtype=np.float)

        dec_bins_to_consider_idx = np.flatnonzero((dec_bins_upper_edge >= dec_min) & (dec_bins_lower_edge <= dec_max))
        #dec_bins_to_consider_idx = np.flatnonzero((upper_edges >= dec_min) & (lower_edges <= dec_max))
        #log.info("decbins before adding the extra bins : %s" %(dec_bins_to_consider_idx))
        #log.info("Decbins to from response file before adding the extra bins : %s" %(dec_bins_to_consider_idx2))

        dec_bins_to_consider_idx = np.append(dec_bins_to_consider_idx, [dec_bins_to_consider_idx[-1] + 1])
        # Add one dec bin to cover the first part
        dec_bins_to_consider_idx = np.insert(dec_bins_to_consider_idx, 0, [dec_bins_to_consider_idx[0] - 1])
        # Rescale bins to be remove dec bins less than 0 and greater than 22
        dec_bins_in_use=dec_bins_to_consider_idx[(dec_bins_to_consider_idx >=0) | (-23>dec_bins_to_consider_idx)]
        log.info("Declination bins to read response file: %s" %(dec_bins_in_use))
        return dec_bins_in_use