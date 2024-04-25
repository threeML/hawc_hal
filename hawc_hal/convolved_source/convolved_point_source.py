from __future__ import division
from builtins import zip
from builtins import object
import os
import collections
import numpy as np

from hawc_hal.psf_fast import PSFInterpolator
from hawc_hal import HAL

from astromodels import use_astromodels_memoization
from astromodels.utils.angular_distance import angular_distance
from threeML.io.logging import setup_logger
log = setup_logger(__name__)
log.propagate = False

from scipy.ndimage import shift


class ConvolvedPointSource(object):

    def __init__(self, source, response, flat_sky_projection, _active_planes):

        self._response = response
        self._flat_sky_projection = flat_sky_projection
        self._active_planes = _active_planes

        # Get name
        self._name = source.name
        self._source = source

        # Get the RA and Dec of our source
        lon = self._source.position.ra.value
        lat = self._source.position.dec.value

        # Get the defined dec bins lower edges
        self._lower_edges = np.array([x[0] for x in response.dec_bins])
        self._upper_edges = np.array([x[-1] for x in response.dec_bins])
        self._centers = np.array([x[1] for x in response.dec_bins])

        #Now find the dec bins we need to consider and pull out the psf's
        if (self._source.position.dec.free):
            dec_min = self._source.position.dec.min_value
            dec_max = self._source.position.dec.max_value
        else: 
            #if we have a fixed source
            dec_min = dec_max = lat

        # Find the dec bins within which the source can move
        self._dec_bins_to_consider_idx = np.flatnonzero( (self._upper_edges >= dec_min) & (self._lower_edges <= dec_max) )
        # Add one dec bin to cover the last part
        if (self._centers[self._dec_bins_to_consider_idx[-1]] < dec_max):
            # If not in very last bin
            if ( self._dec_bins_to_consider_idx[-1] != len(self._centers) - 1 ):
                self._dec_bins_to_consider_idx = np.append(self._dec_bins_to_consider_idx, [self._dec_bins_to_consider_idx[-1] + 1])
        # Add one dec bin to cover the first part
        if (self._centers[self._dec_bins_to_consider_idx[0]] > dec_min):
            # If not in very first bin
            if ( self._dec_bins_to_consider_idx[0] != 0 ):
                self._dec_bins_to_consider_idx = np.insert(self._dec_bins_to_consider_idx, 0, [self._dec_bins_to_consider_idx[0] - 1])

        self._dec_bins_to_consider = [self._response.response_bins[self._centers[x]] for x in self._dec_bins_to_consider_idx]

        log.info("Considering %i dec bins for point source %s" % (len(self._dec_bins_to_consider), self._name))
        log.info("Dec bins are %s" % (self._dec_bins_to_consider_idx) )

        self._decbin_to_idx = {}
        for i, decbin in enumerate(self._dec_bins_to_consider_idx):
            self._decbin_to_idx[ decbin ] = i

        # Prepare somewhere to save convoluters and convolved images for all dec bins and energy bins
        # Convoluters
        self._psf_convolutors = []
        for response_bin in self._dec_bins_to_consider:
            psf_convolutor = collections.OrderedDict()
            for energy_bin_id in self._active_planes:
                psf_convolutor[ energy_bin_id ] = PSFInterpolator( response_bin[ energy_bin_id ].psf, self._flat_sky_projection )
            self._psf_convolutors.append( psf_convolutor )

        # This stores the convoluted sources
        self._this_model_image_norm_conv = []
        for response_bin in self._dec_bins_to_consider:
            this_model_image_norm_conv = collections.OrderedDict( dict.fromkeys( self._active_planes ) )
            self._this_model_image_norm_conv.append( this_model_image_norm_conv )

        # Store weights and Dec bin indices for current position
        self._weights = collections.OrderedDict( dict.fromkeys( self._active_planes ) )
        self._dec_bins_idx = collections.OrderedDict( dict.fromkeys( self._active_planes ) )

        # This stores the convoluted sources
        self._this_model_image_norm_conv_shifted = collections.OrderedDict( dict.fromkeys( self._active_planes ) )

        # Get the initial skymask and energies
        self._energy_centers_keV = self._update_energy_centers( lat )

        # Prepare array for fluxes
        self._all_fluxes = np.zeros( self._energy_centers_keV.shape[0] )

        # This stores the position that was processed during the last step, store a dummy value here for the time being
        self._last_processed_position = collections.OrderedDict( dict.fromkeys( self._active_planes ) )
        for key, value in self._last_processed_position.items():
            self._last_processed_position[ key ] = (0., 0.)

        self._last_processed_position_idx = collections.OrderedDict( dict.fromkeys( self._active_planes ) )
        for key, value in self._last_processed_position_idx.items():
            self._last_processed_position_idx[ key ] = [0, 0]        
       
        # We implement a caching system so that the source flux is evaluated only when strictly needed,
        # because it is the most computationally intense part otherwise.
        self._recompute = True

        # Register callback to keep track of the parameter changes
        self._setup_callbacks(self._parameter_change_callback)

        # This is a simple counter that counts the number of iterations in each energy bin
        self.new_bool = collections.OrderedDict( dict.fromkeys( self._active_planes ) )
        for key, value in self.new_bool.items():
            self.new_bool[ key ] = True


    def _get_relevant_bins( self, dec_current ):

        dec_bins_idx = np.flatnonzero((self._upper_edges >= dec_current) & (self._lower_edges <= dec_current))

        # Add one dec bin to cover the last part
        if (self._centers[dec_bins_idx[-1]] < dec_current):
            # If not in very last bin
            if ( dec_bins_idx[-1] != len(self._centers) - 1 ):
                dec_bins_idx = np.append(dec_bins_idx, [dec_bins_idx[-1] + 1])
            else:
                dec_bins_idx = np.append(dec_bins_idx, [dec_bins_idx[-1]])

        # Add one dec bin to cover the first part
        if (self._centers[dec_bins_idx[0]] > dec_current):
            # If not in very first bin
            if ( dec_bins_idx[0] != 0 ):
                dec_bins_idx = np.insert(dec_bins_idx, 0, [dec_bins_idx[0] - 1])
            else:
                dec_bins_idx = np.insert(dec_bins_idx, 0, [dec_bins_idx[0]])

        assert dec_bins_idx[0] <= dec_bins_idx[1], "Dec bins are in the wrong order!"

        return dec_bins_idx


    def _get_weights( self, dec_bins_idx, energy_bin_id ):
        
        dec_bin1, dec_bin2 = self._dec_bins_to_consider[ 
            self._decbin_to_idx[ dec_bins_idx[0] ] ], self._dec_bins_to_consider[ 
            self._decbin_to_idx[ dec_bins_idx[1] ] ]

        # Get the two response bins to consider
        this_response_bin1, this_response_bin2 = dec_bin1[ energy_bin_id ], dec_bin2[ energy_bin_id ]

        # Figure out which pixels are between the centers of the dec bins we are considering
        c1, c2 = this_response_bin1.declination_center, this_response_bin2.declination_center

        # Compute the interpolation weights for the two responses
        w1 = (self._pix_ctr_coords[1] - c2) / (c1 - c2)
        w2 = (self._pix_ctr_coords[1] - c1) / (c2 - c1)

        return np.array( [w1, w2 ] )


    def _update_energy_centers( self, dec_current):

        # Find central bin for the PSF
        self._central_response_bins = self._response.get_response_dec_bin( dec_current, interpolate=False )

        # Get the energies needed for the computation of the flux
        energy_centers_keV = self._central_response_bins[list(self._central_response_bins.keys())[0]].sim_energy_bin_centers * 1e9

        return energy_centers_keV


    def _parameter_change_callback(self, this_parameter):

        # A parameter has changed, we need to recompute the function.
        # NOTE: we do not recompute it here because if more than one parameter changes at the time (like for example
        # during sampling) we do not want to recompute the function multiple time before we get to the convolution
        # stage. Therefore we will compute the function in the get_source_map method

        #This is a general callback - any parameter changes this gets set to True

        # print("%s has changed" % this_parameter.name)
        self._recompute = True


    def _setup_callbacks(self, callback):

        # Register call back with all free parameters and all linked parameters. If a parameter is linked to another
        # one, the other one might be in a different source, so we add a callback to that one so that we recompute
        # this source when needed.
        for parameter in list(self._source.parameters.values()):

            if parameter.free:
                parameter.add_callback(callback)

            if parameter.has_auxiliary_variable:
                # Add a callback to the auxiliary variable so that when that is changed, we need
                # to recompute the model
                aux_variable, _ = parameter.auxiliary_variable

                aux_variable.add_callback(callback)



    def get_source_map(self, energy_bin_id, tag=None ):

        # We do not use the memoization in astromodels because we are doing caching by ourselves,
        # so the astromodels memoization would turn into 100% cache miss and use a lot of RAM for nothing,
        # given that we are evaluating the function on many points and many energies
        with use_astromodels_memoization(False):

            ra_current, dec_current = self._source.position.ra.value, self._source.position.dec.value

            if self.new_bool[ energy_bin_id ]:
                
                # Only need to do this once
                self.new_bool[ energy_bin_id ] = False

                # Using one of the ordered dicts to check whether this is the first energy bin in the collection as only need to
                # calc some things once for each iteration across all bins
                if ( energy_bin_id == self._active_planes[0] ):

                    # Find the index in RA and Dec of the pixel containing the source and the pixel centre coordinates
                    self._idx = self._flat_sky_projection.wcs.world_to_array_index_values( [[ ra_current, dec_current ]] )[0]
                    self._pix_ctr_coords = self._flat_sky_projection.wcs.array_index_to_world_values( [self._idx] )[0]

                    self._deltaidx = ( self._flat_sky_projection.wcs.world_to_pixel_values( [[ ra_current, dec_current ]] )[0] -
                                       self._flat_sky_projection.wcs.world_to_pixel_values( [self._pix_ctr_coords] )[0] )

                    # Create an empty array, find the active pixel and set the active pixel = 1.
#                    self._this_model_image_square = np.zeros( (self._flat_sky_projection.npix_height, 
#                                                               self._flat_sky_projection.npix_width) )

#                    self._this_model_image_square[ self._idx[1], self._idx[0] ] = 1.

#                    assert np.sum(self._this_model_image_square) == 1., "Yikes! This should have spat out 1. in ConvolvedPointSource"

                # Calculate the PSF in each dec bin for the energy bin, we only need to do this once
                for i in range( len( self._dec_bins_to_consider ) ):

                    self._this_model_image_norm_conv[ i ][ energy_bin_id ] = self._psf_convolutors[ i ][ 
                            energy_bin_id ].point_source_image( self._pix_ctr_coords[0], self._pix_ctr_coords[1], "fast")

                    # The convoluter returns negative pixel values in some cases, this needs to be fixed really, we just botch it here
                    # We don't just truncate at 0. but shift the image, just in case there is gradient information in there that the
                    # minimizer might need.
#                    if ( np.any(self._this_model_image_norm_conv[ i ][ energy_bin_id ] < 0.) ):
                    self._this_model_image_norm_conv[ i ][ energy_bin_id ] = (
                          self._this_model_image_norm_conv[ i ][ energy_bin_id ] +
                          np.abs( np.min( self._this_model_image_norm_conv[ i ][ energy_bin_id ] ) ) )
                    self._this_model_image_norm_conv[ i ][ energy_bin_id ] = (
                           self._this_model_image_norm_conv[ i ][ energy_bin_id ].clip(min=0.) / 
                           np.sum( self._this_model_image_norm_conv[ i ][ energy_bin_id ].clip(min=0.) ) )


                # Get the two decbins that relevant for this points source convolution
                self._dec_bins_idx[ energy_bin_id ] = self._get_relevant_bins( dec_current )

                # Calculate the weight between the two bins
                self._weights[ energy_bin_id ] = self._get_weights( self._dec_bins_idx[ energy_bin_id ], energy_bin_id )

                # Calculate the weighted sum of the convoluted images
                this_model_image_norm_conv_weighted = ( 
                    self._weights[ energy_bin_id ][0] * self._this_model_image_norm_conv[ 
                        self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][0] ] ][ energy_bin_id ] +
                    self._weights[ energy_bin_id ][1] * self._this_model_image_norm_conv[ 
                        self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][1] ] ][ energy_bin_id ] )


                self._this_model_image_norm_conv_shifted[ energy_bin_id ] =  shift( this_model_image_norm_conv_weighted, 
                    self._deltaidx[::-1], order=1, mode='grid-constant', cval=0.0, prefilter=True )

                this_model_image_norm_conv = self._this_model_image_norm_conv_shifted[ energy_bin_id ]

                # Store the last processed position
                self._last_processed_position[ energy_bin_id ] = ( ra_current, dec_current )

                # Store the indices of the position in the RA/Dec grid
                self._last_processed_position_idx[ energy_bin_id ] = self._idx

            # same position so don't need to update the convolution
            elif ( ( ra_current, dec_current ) == self._last_processed_position[ energy_bin_id ] ):

                this_model_image_norm_conv = self._this_model_image_norm_conv_shifted[ energy_bin_id ]

            # Position has changed, so we just shift the convolved source in the image. We can do this at the moment because we are 
            # using only a single psf in the ROI, to improve this, for large ROIs we should really update the PSF and run the first 
            # item in the structure
            else:

                if ( energy_bin_id == self._active_planes[0] ):

                    # Find the new pixel and its centre coordinates
                    self._idx = self._flat_sky_projection.wcs.world_to_array_index_values( [[ ra_current, dec_current ]] )[0]
                    self._pix_ctr_coords = self._flat_sky_projection.wcs.array_index_to_world_values( [self._idx] )[0]

                    # Determine the difference in position and save this for the other energy bins
                    self._deltaidx = ( self._flat_sky_projection.wcs.world_to_pixel_values( [[ ra_current, dec_current ]] )[0] -
                                       self._last_processed_position_idx[ energy_bin_id ] )

                # get the two Dec bins between which to interpolate the PSF, get the weights and calculate the interpolates image
                self._dec_bins_idx[ energy_bin_id ] = self._get_relevant_bins( dec_current )

                self._weights[ energy_bin_id ] = self._get_weights( self._dec_bins_idx[ energy_bin_id ], energy_bin_id )

                this_model_image_norm_conv_weighted = ( 
                    self._weights[ energy_bin_id ][0] * self._this_model_image_norm_conv[ 
                        self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][0] ] ][ energy_bin_id ] +
                    self._weights[ energy_bin_id ][1] * self._this_model_image_norm_conv[ 
                        self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][1] ] ][ energy_bin_id ] )

                # Shift the image with sub-pixel precision
                this_model_image_norm_conv =  shift( this_model_image_norm_conv_weighted, 
                    self._deltaidx[::-1], order=1, mode='grid-constant', cval=0.0, prefilter=True )

            # If we need to recompute the flux, let's do it
            if self._recompute:

                #Check that dec is still between decbin centres, if not need to calculate dec bins again

                if ( dec_current != self._last_processed_position[ energy_bin_id ][1] ):

                    self._energy_centers_keV = self._update_energy_centers( dec_current )

                # Recompute the fluxes for the pixels that are covered by this source, we do this no matter what
                self._all_fluxes = self._source( self._energy_centers_keV, tag=tag )   
                                                                  # 1 / (keV cm^2 s rad^2)

                # We don't need to recompute the function anymore until a parameter changes so reset the callback to False
                self._recompute = False


            # Loop over the Dec bins that cover this source and compute the expected flux, interpolating between
            # two dec bins for each point

            dec_bin1, dec_bin2 = self._dec_bins_to_consider[ 
                self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][0] ] ], self._dec_bins_to_consider[ 
                self._decbin_to_idx[ self._dec_bins_idx[ energy_bin_id ][1] ] ]

            # Get the two response bins to consider
            this_response_bin1, this_response_bin2 = dec_bin1[energy_bin_id], dec_bin2[energy_bin_id]

            # Reweight the spectrum separately for the two bins
            # NOTE: the scale is the same because the sim_differential_photon_fluxes are the same (the simulation
            # used to make the response used the same spectrum for each bin). What changes between the two bins
            # is the observed signal per bin (the .sim_signal_events_per_bin member)
            scale = self._all_fluxes / this_response_bin1.sim_differential_photon_fluxes

            # This scales the unit PSF image to the actual flux
            factor = (self._weights[ energy_bin_id ][0] * np.sum(scale * this_response_bin1.sim_signal_events_per_bin) +
                      self._weights[ energy_bin_id ][1] * np.sum(scale * this_response_bin2.sim_signal_events_per_bin)) * 1e9

            this_model_image_conv = this_model_image_norm_conv * factor
           
            return this_model_image_conv

