import numpy as np
import sys
from astromodels import PointSource, use_astromodels_memoization
from threeML.exceptions.custom_exceptions import custom_warnings
from psf_fast import PSFInterpolator, PSFConvolutor


import matplotlib.pyplot as plt
from matplotlib import colors


# Conversion factor between deg^2 and rad^2
deg2_to_rad2 = 0.00030461741978670857


class ConvolvedPointSource(object):

    def __init__(self, source, response, flat_sky_projections):

        assert isinstance(source, PointSource)

        self._source = source

        # Get name
        self._name = self._source.name

        self._response = response

        self._flat_sky_projections = flat_sky_projections

        # This will store the position of the source
        # right now, let's use a fake value so that at the first iteration the source maps will be filled
        # (see get_expected_signal_per_transit)
        self._last_processed_position = (-999, -999)

        self._response_energy_bins = None
        self._psf_interpolators = None

    @property
    def name(self):

        return self._name

    def _update_dec_bins(self, dec_src):

        self._response_energy_bins, _ = self._response.get_response_dec_bin(dec_src)

        # Setup the PSF interpolators
        self._psf_interpolators = map(lambda (response_bin, flat_sky_proj): PSFInterpolator(response_bin.psf,
                                                                                            flat_sky_proj),
                                      zip(self._response_energy_bins, self._flat_sky_projections))

    def get_source_map(self, response_bin_id, tag=None):

        # Get current point source position
        # NOTE: this might change if the point source position is free during the fit,
        # that's why it is here
        ra_src, dec_src = self._source.position.ra.value, self._source.position.dec.value

        if (ra_src, dec_src) != self._last_processed_position:

            # Position changed (or first iteration), let's update the dec bins
            self._update_dec_bins(dec_src)

            self._last_processed_position = (ra_src, dec_src)

        # Get the current response bin
        response_energy_bin = self._response_energy_bins[response_bin_id]
        psf_interpolator = self._psf_interpolators[response_bin_id]

        # Get the PSF image
        # This is cached inside the PSF class, so that if the position doesn't change this line
        # is very fast
        this_map = psf_interpolator.point_source_image(ra_src, dec_src)

        # Check that the point source is contained in the ROI, if not print a warning
        if not np.isclose(this_map.sum(), 1.0, rtol=1e-2):

            custom_warnings.warn("PSF for source %s is not entirely contained in ROI" % self._name)

        # Compute the fluxes from the spectral function at the same energies as the simulated function
        energy_centers_keV = response_energy_bin.sim_energy_bin_centers * 1e9  # astromodels expects energies in keV

        # This call needs to be here because the parameters of the model might change,
        # for example during a fit

        source_diff_spectrum = self._source(energy_centers_keV, tag=tag)

        # Transform from keV^-1 cm^-2 s^-1 to TeV^-1 cm^-2 s^-1
        source_diff_spectrum *= 1e9

        # Re-weight the detected counts
        scale = source_diff_spectrum / response_energy_bin.sim_differential_photon_fluxes

        # Now return the map multiplied by the scale factor
        return np.sum(scale * response_energy_bin.sim_signal_events_per_bin) * this_map


class ConvolvedExtendedSource2D(object):

    def __init__(self, source, response, flat_sky_projections):

        assert source.spatial_shape.n_dim == 2, "Use the ConvolvedExtendedSource3D for this source"

        self._response = response
        self._flat_sky_projections = flat_sky_projections

        # Get name
        self._name = source.name

        self._source = source

        # Find out the response bins we need to consider for this extended source

        # # Get the footprint (i..e, the coordianates of the 4 points limiting the projections)
        # (ra1, dec1), (ra2, dec2), (ra3, dec3), (ra4, dec4) = flat_sky_projections.wcs.calc_footprint()
        # # Figure out maximum and minimum declination to be covered
        # dec_min = min([dec1, dec2, dec3, dec4])
        # dec_max = max([dec1, dec2, dec3, dec4])
        #
        # # Get the defined dec bins lower edges
        # lower_edges = np.array(map(lambda x:x[0], response.dec_bins))
        # upper_edges = np.array(map(lambda x:x[-1], response.dec_bins))
        # centers = np.array(map(lambda x:x[1], response.dec_bins))
        #
        # dec_bins_to_consider_idx = np.flatnonzero((lower_edges >= dec_min) & (upper_edges <= dec_max))
        #
        # self._dec_bins_to_consider = map(lambda x: response.response_bins[centers[x]], dec_bins_to_consider_idx)
        #
        # print("Considering %i dec bins for extended source %s" % (len(self._dec_bins_to_consider),
        #                                                           self._name))

        # Setup fast convolution
        self._central_response_bins, _ = response.get_response_dec_bin(flat_sky_projections[0].dec_center)
        self._psf_convolutors = map(lambda (response_bin, flat_sky_proj): PSFConvolutor(response_bin.psf,
                                                                                        flat_sky_proj),
                                    zip(self._central_response_bins, self._flat_sky_projections))

        # Set up the caching
        self._former_values = {}
        self._spectral_part = {}
        self._spatial_part = {}

        # Register call back
        for parameter in self._source.free_parameters.values():

            parameter.add_callback(self._parameter_change_callback)

        # Now make a list of parameters for the spectral shape.
        self._spectral_free_parameters = []

        for component in self._source.components.values():

            for parameter in component.shape.free_parameters.values():

                self._spectral_free_parameters.append(parameter)

    def _parameter_change_callback(self, this_parameter):

        this_name = this_parameter.name
        this_value = this_parameter.value

        # Check whether this parameter has changed values with respect to last iteration
        if this_value != self._former_values.get(this_name):

            if this_parameter in self._spectral_free_parameters:

                self._spectral_part = {}

            else:

                self._spatial_part = {}

        self._former_values[this_name] = this_value

    def _compute_expected_spectral_signal(self, response_bin):

        energy_centers_keV = response_bin.sim_energy_bin_centers * 1e9  # astromodels expects energies in keV

        # First get the differential flux from the spectral components (most likely there is only one component,
        # but let's do it so it can generalize to multi-component sources)

        results = [component.shape(energy_centers_keV) for component in self._source.components.values()]

        this_diff_flux = np.sum(results, 0)

        scale = this_diff_flux * 1e9 / response_bin.sim_differential_photon_fluxes

        expected_signal = np.sum(scale * response_bin.sim_signal_events_per_bin)

        return expected_signal

    def get_source_map(self, energy_bin_id, tag=None):

        # We do not use the memoization in astromodels because we are doing caching by ourselves,
        # so the astromodels memoization would turn into 100% cache miss
        with use_astromodels_memoization(False):

            # Spectral part: first we try to see if we have it already in the cache
            spectral_part = self._spectral_part.get(energy_bin_id)

            if spectral_part is None:

                # Cache miss

                # Get current response bin

                response_bin = self._central_response_bins[energy_bin_id]

                # Compute the spectral part
                spectral_part = self._compute_expected_spectral_signal(response_bin)

                # Memorize hoping to reuse it next time
                self._spectral_part[energy_bin_id] = spectral_part

            # Spatial part: first we try to see if we have it already in the cache
            spatial_part = self._spatial_part.get(energy_bin_id)

            if spatial_part is None:

                # Cache miss, we need to reconvolute

                # Get flat sky projection
                this_flat_sky_p = self._flat_sky_projections[energy_bin_id]

                # We substitute integration with a discretization, which will work
                # well if the size of the pixel of the flat sky grid is small enough compared to the features of the
                # extended source
                brightness = self._source.spatial_shape(this_flat_sky_p.ras, this_flat_sky_p.decs)  # 1 / rad^2

                pixel_area_rad2 = this_flat_sky_p.project_plane_pixel_area * deg2_to_rad2

                integrated_flux = brightness * pixel_area_rad2

                # Now convolve with psf
                this_model_image = integrated_flux.reshape((this_flat_sky_p.npix_height, this_flat_sky_p.npix_width))
                spatial_part = self._psf_convolutors[energy_bin_id].extended_source_image(this_model_image)

                # Memorize hoping to reuse it next time
                self._spatial_part[energy_bin_id] = spatial_part

        return spectral_part * spatial_part


class ConvolvedExtendedSource3D(object):

    def __init__(self, source, response, flat_sky_projections):

        self._response = response
        self._flat_sky_projections = flat_sky_projections

        # Get name
        self._name = source.name

        self._source = source

        # Find out the response bins we need to consider for this extended source

        # # Get the footprint (i..e, the coordianates of the 4 points limiting the projections)
        # (ra1, dec1), (ra2, dec2), (ra3, dec3), (ra4, dec4) = flat_sky_projections.wcs.calc_footprint()
        # # Figure out maximum and minimum declination to be covered
        # dec_min = min([dec1, dec2, dec3, dec4])
        # dec_max = max([dec1, dec2, dec3, dec4])
        #
        # # Get the defined dec bins lower edges
        # lower_edges = np.array(map(lambda x:x[0], response.dec_bins))
        # upper_edges = np.array(map(lambda x:x[-1], response.dec_bins))
        # centers = np.array(map(lambda x:x[1], response.dec_bins))
        #
        # dec_bins_to_consider_idx = np.flatnonzero((lower_edges >= dec_min) & (upper_edges <= dec_max))
        #
        # self._dec_bins_to_consider = map(lambda x: response.response_bins[centers[x]], dec_bins_to_consider_idx)
        #
        # print("Considering %i dec bins for extended source %s" % (len(self._dec_bins_to_consider),
        #                                                           self._name))

        # Setup fast convolution
        self._central_response_bins, _ = response.get_response_dec_bin(flat_sky_projections[0].dec_center)
        self._psf_convolutors = map(lambda (response_bin, flat_sky_proj): PSFConvolutor(response_bin.psf,
                                                                                        flat_sky_proj),
                                    zip(self._central_response_bins, self._flat_sky_projections))

        # Set up the caching
        self._last_parameter_values = {}
        self._last_convolved_image = {}
        self._convolution_renorm = 1.0

        # Register call back
        for parameter in self._source.free_parameters.values():

            parameter.add_callback(self._parameter_change_callback)

    def _convolution_cache_reset(self):

        self._last_convolved_image = {}
        self._convolution_renorm = 1.0

    def _parameter_change_callback(self, this_parameter):

        this_name = this_parameter.name
        this_value = this_parameter.value

        # Check whether this parameter has changed values with respect to last iteration

        last_value = self._last_parameter_values.get(this_name)

        if last_value != this_value:

            # Yes, this parameter changed value.

            # If this parameter is a normalization, we will just renorm the existing images,
            # unless another parameter has been changed
            if this_parameter.is_normalization and last_value is not None:

                self._convolution_renorm = this_value / self._last_parameter_values[this_name]

            else:

                # A normal parameter.
                # This parameter has changed value. Need to re-convolute

                self._convolution_cache_reset()

        # Take note of the current value
        self._last_parameter_values[this_name] = this_value

    def get_source_map(self, energy_bin_id, tag=None):

        last_convolved_image = self._last_convolved_image.get(energy_bin_id)

        if last_convolved_image is not None:

            # We have an available image for the same energy bin, and all parameters
            # (except maybe the normalization) did not change with respect to the last
            # iteration. let's use that instead of recomputing a new image
            self._last_convolved_image[energy_bin_id] = last_convolved_image * self._convolution_renorm

            # No need to reconvolute, we already have the right image
            return self._last_convolved_image[energy_bin_id]

        # If we are here, we need to reconvolute
        this_flat_sky_p = self._flat_sky_projections[energy_bin_id]

        response_bin = self._central_response_bins[energy_bin_id]

        energy_centers_keV = response_bin.sim_energy_bin_centers * 1e9  # astromodels expects energies in keV

        # Slow version: this is a 3d function and the spectrum is different in each point

        this_diff_flux = self._source(this_flat_sky_p.ras, this_flat_sky_p.decs, energy_centers_keV)

        pixel_area_rad2 = this_flat_sky_p.project_plane_pixel_area * deg2_to_rad2

        # Reweight
        scale = this_diff_flux * 1e9 / response_bin.sim_differential_photon_fluxes * pixel_area_rad2

        this_model_image = np.sum(scale * response_bin.sim_signal_events_per_bin, axis=1)

        # Now convolve with psf
        this_model_image = this_model_image.reshape((this_flat_sky_p.npix_height, this_flat_sky_p.npix_width))

        convolved_image = self._psf_convolutors[energy_bin_id].extended_source_image(this_model_image)

        # Save it for next time, we'll reuse it if parameters did not change
        self._last_convolved_image[energy_bin_id] = convolved_image

        return convolved_image