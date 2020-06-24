from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from threeML import *
import numpy as np
import warnings
from hawc_hal.HAL import HAL
from hawc_hal.region_of_interest import HealpixConeROI

from astropy.wcs import WCS
from astropy.coordinates.angle_utilities import angular_separation
from astropy.io import fits as pyfits

import matplotlib.pyplot as plt

from threeML.parallel.parallel_client import ParallelClient, is_parallel_computation_active
from threeML.io.progress_bar import progress_bar
from threeML.io.fits_file import FITSFile

crab_diff_flux_at_1_TeV = 2.65e-11 / (u.TeV * u.cm**2 * u.s)


class ParallelTSmap(object):
    
    def __init__(self, maptree, response, ra_c, dec_c,  xsize, ysize, pix_scale, projection = 'AIT', roi_radius=3.0):

        # Create a new WCS object so that we can compute the appropriare R.A. and Dec
        # where we need to compute the TS
        self._wcs = wcs.WCS(naxis=2)
        
        # The +1 is because the CRPIX should be in FORTRAN indexing, which starts at +1
        self._wcs.wcs.crpix = [xsize / 2.0, ysize / 2.0]
        self._wcs.wcs.cdelt = [-pix_scale, pix_scale]
        self._wcs.wcs.crval = [ra_c, dec_c]
        self._wcs.wcs.ctype = ["RA---%s" % projection, "DEC--%s" % projection]
        
        self._ra_c = ra_c
        self._dec_c = dec_c
        
        self._mtfile = maptree
        self._rsfile = response
        
        self._points = []
        
        # It is important that dec is the first one because the PSF for a Dec bin_name
        # is cached within one engine
        
        max_d = 0
        
        for idec in range(ysize):
            
            for ira in range(xsize):
                 
                 this_ra, this_dec = self._wcs.wcs_pix2world(ira, idec, 0)
                 
                 self._points.append((this_ra, this_dec))
                 
                 d = angular_separation(*np.deg2rad((this_ra, this_dec, ra_c, dec_c)))
                 
                 if d > max_d:
                     
                     max_d = d
                 
        print("Maximum distance from center: %.3f deg" % np.rad2deg(max_d))  
        
        # We keep track of how many ras we have so that when running in parallel all
        # the ras will run on the same engine with the same dec, maximizing the use
        # of the cache and minimizing the memory footprint
        
        self._n_ras = xsize
        self._n_decs = ysize
        
        self._roi_radius = float(roi_radius)

        roi = HealpixConeROI(self._roi_radius, ra=ra_c, dec=dec_c)
        
        self._llh = HAL("HAWC", self._mtfile, self._rsfile, roi)
        self._llh.set_active_measurements(1,9)
        
        # Make a fit with no source to get the likelihood for the null hypothesis
        model = self.get_model(0)
        model.TestSource.spectrum.main.shape.K = model.TestSource.spectrum.main.shape.K.min_value
        
        self._llh.set_model(model)
        
        self._like0 = self._llh.get_log_like()
        
        # We will fill this with the maximum of the TS map
        
        self._max_ts = None
    
    def get_data(self, interval_id):
        
        datalist = DataList(self._llh)
        
        return datalist
     
    def get_model(self, interval_id):
        
        spectrum = Powerlaw()
        
        this_ra = self._points[interval_id][0]
        this_dec = self._points[interval_id][1]
        
        this_source = PointSource("TestSource", ra=this_ra, dec=this_dec, spectral_shape=spectrum)
        
        spectrum.piv = 1 * u.TeV
        spectrum.piv.fix = True
                
        # Start from a flux 1/10 of the Crab
        spectrum.K = crab_diff_flux_at_1_TeV
        spectrum.K.bounds = (1e-30, 1e-18)
        spectrum.index = -2.63
        spectrum.index.fix = False
        
        model1 = Model(this_source)
    
        return model1
    
    def worker(self, interval_id):
        
        model = self.get_model(interval_id)
        data = self.get_data(interval_id)
        
        jl = JointLikelihood(model, data)
        jl.set_minimizer("ROOT")
        par, like = jl.fit(quiet=True)
        
        return like['-log(likelihood)']['HAWC']

    
    def go(self):
                
        if is_parallel_computation_active():
        
            client = ParallelClient()
            
            if self._n_decs % client.get_number_of_engines() != 0:
                
                warnings.warn("The number of Dec bands is not a multiple of the number of engine. Make it so for optimal performances.", RuntimeWarning)
            
            res = client.execute_with_progress_bar(self.worker, list(range(len(self._points))), chunk_size=self._n_ras)
        
        else:
            
            n_points = len(self._points)
            
            with progress_bar(n_points) as p:
                
                res = np.zeros(n_points)
                
                for i, point in enumerate(self._points):
                    
                    res[i] = self.worker(i)
                    
                    p.increase()
        
        TS = 2 * (-np.array(res) - self._like0)
        
        #self._debug_map = {k:v for v,k in zip(self._points, TS)}
        
        # Get maximum of TS
        idx = TS.argmax()
        self._max_ts = (TS[idx], self._points[idx])
        
        print("Maximum TS is %.2f at (R.A., Dec) = (%.3f, %.3f)" % (self._max_ts[0], self._max_ts[1][0], self._max_ts[1][1]))
         
        self._ts_map = TS.reshape(self._n_decs, self._n_ras)
        
        return self._ts_map
    
    @property
    def maximum_of_map(self):
        
        return self._max_ts
    
    def to_fits(self, filename, overwrite=True):
        
        primary_hdu = pyfits.PrimaryHDU(data=self._ts_map, header=self._wcs.to_header())
        
        primary_hdu.writeto(filename, overwrite=overwrite)
    
    def plot(self):
        
        # Draw TS map
        
        fig, ax = plt.subplots(subplot_kw = {'projection': self._wcs})

        plt.imshow(self._ts_map, origin='lower', interpolation='none')
        
        # Draw colorbar
        
        cbar = plt.colorbar(format="%.1f")
        
        cbar.set_label("TS")
        
        # Plot 1, 2 and 3 sigma contour
        _ = plt.contour(self._ts_map, origin='lower', 
                        levels=self._ts_map.max() - np.array([4.605, 7.378, 9.210][::-1]),
                        colors=['black', 'blue','red'])
        
        # Access the first WCS coordinate
        ra = ax.coords[0]
        dec = ax.coords[1]
        
        # Set the format of the tick labels
        ra.set_major_formatter('d.ddd')
        dec.set_major_formatter('d.ddd')
        
        plt.xlabel('R.A. (J2000)')
        plt.ylabel('Dec. (J2000)')
        
        # Overlay the center
        
        ax.scatter([self._ra_c],[self._dec_c], transform=ax.get_transform('world'), marker='o')
        
        # Overlay the maximum TS
        ra_max, dec_max = self._max_ts[1]
        
        ax.scatter([ra_max],[dec_max], transform=ax.get_transform('world'), marker='x')
        
        return fig

