from __future__ import division
from builtins import zip
from past.utils import old_div
from builtins import object
import numpy as np

from astromodels import use_astromodels_memoization
from threeML.io.logging import setup_logger
log = setup_logger(__name__)
log.propagate = False

class Index_search(object):

    def __init__(self, source, flat_sky_projection):

        self._flat_sky_projection = flat_sky_projection

        # Get name
        self._name = source.name

        self._source = source
 
        # Find out the response bins we need to consider for this extended source

        # # Get the footprint (i..e, the coordinates of the 4 points limiting the projections)
        (ra1, dec1), (ra2, dec2), (ra3, dec3), (ra4, dec4) = flat_sky_projection.wcs.calc_footprint()

        (lon_start, lon_stop), (lat_start, lat_stop) = source.get_boundaries()

        # Figure out maximum and minimum declination to be covered
        dec_min = max(min([dec1, dec2, dec3, dec4]), lat_start)
        dec_max = min(max([dec1, dec2, dec3, dec4]), lat_stop)

# Get the defined dec bins lower edges
        #lower_edges = np.array([x[0] for x in response.dec_bins])
        lower_edges = [-37.5 -32.5 -27.5 -22.5 -17.5 -12.5  -7.5  -2.5  2.5  7.5  12.5  17.5 22.5  27.5  32.5  37.5  42.5  47.5  52.5  57.5  62.5  67.5  72.5]
        #upper_edges = np.array([x[-1] for x in response.dec_bins])
        upper_edges = [-32.5 -27.5 -22.5 -17.5 -12.5  -7.5  -2.5   2.5   7.5  12.5  17.5  22.5  27.5  32.5  37.5  42.5  47.5  52.5  57.5  62.5  67.5  72.5  77.5]
        #centers = np.array([x[1] for x in response.dec_bins])
        centers = [-35. -30. -25. -20. -15. -10.  -5.   0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.  60.  65.  70.  75.]
        log.info("Lower edge: %s" %(lower_edges))
        log.info("upper edge: %s" %(upper_edges))
        log.info("centers : %s" %(centers))
        dec_bins_to_consider_idx = np.flatnonzero((upper_edges >= dec_min) & (lower_edges <= dec_max))

        
