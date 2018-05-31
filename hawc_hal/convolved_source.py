import numpy as np
from astromodels import PointSource, use_astromodels_memoization
from threeML.exceptions.custom_exceptions import custom_warnings
from psf_fast import PSFInterpolator


def _select_with_wrap_around(arr, start, stop, wrap=(360, 0)):

    if start > stop:

        # This can happen if for instance lon_start = 280 and lon_stop = 10, which means we
        # should keep between 280 and 360 and then between 360 to 10
        idx = ((arr >= stop) & (arr <= wrap[0])) | ((arr >= wrap[1]) & (arr <= start))

    else:

        idx = (arr >= start) & (arr <= stop)

    return idx

# Conversion factor between deg^2 and rad^2
deg2_to_rad2 = 0.00030461741978670857


class ConvolvedPointSource(object):

    def __init__(self, source, response, flat_sky_projection):

        assert isinstance(source, PointSource)

        self._source = source

        # Get name
        self._name = self._source.name

        self._response = response

        self._flat_sky_projection = flat_sky_projection

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

        if abs(dec_src - self._last_processed_position[1]) > 0.1:

            # Source moved by more than 0.1 deg, let's recompute the response and the PSF
            self._response_energy_bins = self._response.get_response_dec_bin(dec_src, interpolate=True)

            # Setup the PSF interpolators
            self._psf_interpolators = map(lambda response_bin: PSFInterpolator(response_bin.psf,
                                                                               self._flat_sky_projection),
                                          self._response_energy_bins)

        # for i, psf_i in enumerate(self._psf_interpolators):
        #
        #     print("PSF %i: %.3f, %.3f" % (i, psf_i._psf.truncation_radius, psf_i._psf.kernel_radius))

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
        map_sum = this_map.sum()
        if not np.isclose(map_sum, 1.0, rtol=1e-2):

            custom_warnings.warn("PSF for source %s is not entirely contained "
                                 "in ROI for response bin %s. Fraction is %.2f instead of 1.0" % (self._name,
                                                                                                  response_bin_id,
                                                                                                  map_sum))

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


class ConvolvedExtendedSource(object):

    def __init__(self, source, response, flat_sky_projection):

        self._response = response
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
        lower_edges = np.array(map(lambda x: x[0], response.dec_bins))
        upper_edges = np.array(map(lambda x: x[-1], response.dec_bins))
        centers = np.array(map(lambda x: x[1], response.dec_bins))

        dec_bins_to_consider_idx = np.flatnonzero((upper_edges >= dec_min) & (lower_edges <= dec_max))

        # Wrap the selection so we have always one bin before and one after.
        # NOTE: we assume that the ROI do not overlap with the very first or the very last dec bin
        # Add one dec bin to cover the last part
        dec_bins_to_consider_idx = np.append(dec_bins_to_consider_idx, [dec_bins_to_consider_idx[-1] + 1])
        # Add one dec bin to cover the first part
        dec_bins_to_consider_idx = np.insert(dec_bins_to_consider_idx, 0, [dec_bins_to_consider_idx[0] - 1])

        self._dec_bins_to_consider = map(lambda x: response.response_bins[centers[x]], dec_bins_to_consider_idx)

        print("Considering %i dec bins for extended source %s" % (len(self._dec_bins_to_consider),
                                                                  self._name))

        # Find central bin for the PSF

        dec_center = (lat_start + lat_stop) / 2.0
        #
        self._central_response_bins = response.get_response_dec_bin(dec_center, interpolate=False)

        print("Central bin is bin at Declination = %.3f" % dec_center)

        # Take note of the pixels within the flat sky projection that actually need to be computed. If the extended
        # source is significantly smaller than the flat sky projection, this gains a substantial amount of time

        idx_lon = _select_with_wrap_around(self._flat_sky_projection.ras, lon_start, lon_stop, (360, 0))
        idx_lat = _select_with_wrap_around(self._flat_sky_projection.decs, lat_start, lat_stop, (90, -90))

        self._active_flat_sky_mask = (idx_lon & idx_lat)

        assert np.sum(self._active_flat_sky_mask) > 0, "Mismatch between source %s and ROI" % self._name

        # Get the energies needed for the computation of the flux
        self._energy_centers_keV = self._central_response_bins[0].sim_energy_bin_centers * 1e9

        # Prepare array for fluxes
        self._all_fluxes = np.zeros((self._flat_sky_projection.ras.shape[0],
                                     self._energy_centers_keV.shape[0]))

    def _setup_callbacks(self, callback):

        # Register call back with all free parameters and all linked parameters. If a parameter is linked to another
        # one, the other one might be in a different source, so we add a callback to that one so that we recompute
        # this source when needed.
        for parameter in self._source.parameters.values():

            if parameter.free:
                parameter.add_callback(callback)

            if parameter.has_auxiliary_variable():
                # Add a callback to the auxiliary variable so that when that is changed, we need
                # to recompute the model
                aux_variable, _ = parameter.auxiliary_variable

                aux_variable.add_callback(callback)

    def get_source_map(self, energy_bin_id, tag=None):

        raise NotImplementedError("You need to implement this in derived classes")


class ConvolvedExtendedSource3D(ConvolvedExtendedSource):

    def __init__(self, source, response, flat_sky_projection):

        super(ConvolvedExtendedSource3D, self).__init__(source, response, flat_sky_projection)

        # We implement a caching system so that the source flux is evaluated only when strictly needed,
        # because it is the most computationally intense part otherwise.
        self._recompute_flux = True

        # Register callback to keep track of the parameter changes
        self._setup_callbacks(self._parameter_change_callback)

    def _parameter_change_callback(self, this_parameter):

        # A parameter has changed, we need to recompute the function.
        # NOTE: we do not recompute it here because if more than one parameter changes at the time (like for example
        # during sampling) we do not want to recompute the function multiple time before we get to the convolution
        # stage. Therefore we will compute the function in the get_source_map method

        # print("%s has changed" % this_parameter.name)
        self._recompute_flux = True

    def get_source_map(self, energy_bin_id, tag=None):

        # We do not use the memoization in astromodels because we are doing caching by ourselves,
        # so the astromodels memoization would turn into 100% cache miss and use a lot of RAM for nothing,
        # given that we are evaluating the function on many points and many energies
        with use_astromodels_memoization(False):

            # If we need to recompute the flux, let's do it
            if self._recompute_flux:

                # print("recomputing %s" % self._name)

                # Recompute the fluxes for the pixels that are covered by this extended source
                self._all_fluxes[self._active_flat_sky_mask, :] = self._source(
                                                            self._flat_sky_projection.ras[self._active_flat_sky_mask],
                                                            self._flat_sky_projection.decs[self._active_flat_sky_mask],
                                                            self._energy_centers_keV)  # 1 / (keV cm^2 s rad^2)

                # We don't need to recompute the function anymore until a parameter changes
                self._recompute_flux = False

            # Now compute the expected signal

            pixel_area_rad2 = self._flat_sky_projection.project_plane_pixel_area * deg2_to_rad2

            this_model_image = np.zeros(self._all_fluxes.shape[0])

            # Loop over the Dec bins that cover this source and compute the expected flux, interpolating between
            # two dec bins for each point

            for dec_bin1, dec_bin2 in zip(self._dec_bins_to_consider[:-1], self._dec_bins_to_consider[1:]):

                # Get the two response bins to consider
                this_response_bin1 = dec_bin1[energy_bin_id]
                this_response_bin2 = dec_bin2[energy_bin_id]

                # Figure out which pixels are between the centers of the dec bins we are considering
                c1, c2 = this_response_bin1.declination_center, this_response_bin2.declination_center

                idx = (self._flat_sky_projection.decs >= c1) & (self._flat_sky_projection.decs < c2) & \
                      self._active_flat_sky_mask

                # Reweight the spectrum separately for the two bins
                # NOTE: the scale is the same because the sim_differential_photon_fluxes are the same (the simulation
                # used to make the response used the same spectrum for each bin). What changes between the two bins
                # is the observed signal per bin (the .sim_signal_events_per_bin member)
                scale = (self._all_fluxes[idx, :] * pixel_area_rad2) / this_response_bin1.sim_differential_photon_fluxes

                # Compute the interpolation weights for the two responses
                w1 = (self._flat_sky_projection.decs[idx] - c2) / (c1 - c2)
                w2 = (self._flat_sky_projection.decs[idx] - c1) / (c2 - c1)

                this_model_image[idx] = (w1 * np.sum(scale * this_response_bin1.sim_signal_events_per_bin, axis=1) +
                                         w2 * np.sum(scale * this_response_bin2.sim_signal_events_per_bin, axis=1)) * \
                                        1e9

            # Reshape the flux array into an image
            this_model_image = this_model_image.reshape((self._flat_sky_projection.npix_height,
                                                         self._flat_sky_projection.npix_width)).T

            return this_model_image


class ConvolvedExtendedSource2D(ConvolvedExtendedSource3D):

    pass

    # def __init__(self, source, response, flat_sky_projection):
    #
    #     assert source.spatial_shape.n_dim == 2, "Use the ConvolvedExtendedSource3D for this source"
    #
    #     super(ConvolvedExtendedSource2D, self).__init__(source, response, flat_sky_projection)
    #
    #     # Set up the caching
    #     self._spectral_part = {}
    #     self._spatial_part = None
    #
    #     # Now make a list of parameters for the spectral shape.
    #     self._spectral_parameters = []
    #     self._spatial_parameters = []
    #
    #     for component in self._source.components.values():
    #
    #         for parameter in component.shape.parameters.values():
    #
    #             self._spectral_parameters.append(parameter.path)
    #
    #             # If there are links, we need to keep track of them
    #             if parameter.has_auxiliary_variable():
    #
    #                 aux_variable, _ = parameter.auxiliary_variable
    #
    #                 self._spectral_parameters.append(aux_variable.path)
    #
    #     for parameter in self._source.spatial_shape.parameters.values():
    #
    #         self._spatial_parameters.append(parameter.path)
    #
    #         # If there are links, we need to keep track of them
    #         if parameter.has_auxiliary_variable():
    #
    #             aux_variable, _ = parameter.auxiliary_variable
    #
    #             self._spatial_parameters.append(aux_variable.path)
    #
    #     self._setup_callbacks(self._parameter_change_callback)
    #
    # def _parameter_change_callback(self, this_parameter):
    #
    #     if this_parameter.path in self._spectral_parameters:
    #
    #         # A spectral parameters. Need to recompute the spectrum
    #         self._spectral_part = {}
    #
    #     else:
    #
    #         # A spatial parameter, need to recompute the spatial shape
    #         self._spatial_part = None
    #
    # def _compute_response_scale(self):
    #
    #     # First get the differential flux from the spectral components (most likely there is only one component,
    #     # but let's do it so it can generalize to multi-component sources)
    #
    #     results = [component.shape(self._energy_centers_keV) for component in self._source.components.values()]
    #
    #     this_diff_flux = np.sum(results, 0)
    #
    #     # NOTE: the sim_differential_photon_fluxes are the same for every bin, so it doesn't matter which
    #     # bin I read them from
    #
    #     scale = this_diff_flux * 1e9 / self._central_response_bins[0].sim_differential_photon_fluxes
    #
    #     return scale
    #
    # def get_source_map(self, energy_bin_id, tag=None):
    #
    #     # We do not use the memoization in astromodels because we are doing caching by ourselves,
    #     # so the astromodels memoization would turn into 100% cache miss
    #     with use_astromodels_memoization(False):
    #
    #         # Spatial part: first we try to see if we have it already in the cache
    #
    #         if self._spatial_part is None:
    #
    #             # Cache miss, we need to recompute
    #
    #             # We substitute integration with a discretization, which will work
    #             # well if the size of the pixel of the flat sky grid is small enough compared to the features of the
    #             # extended source
    #             brightness = self._source.spatial_shape(self._flat_sky_projection.ras, self._flat_sky_projection.decs)  # 1 / rad^2
    #
    #             pixel_area_rad2 = self._flat_sky_projection.project_plane_pixel_area * deg2_to_rad2
    #
    #             integrated_flux = brightness * pixel_area_rad2
    #
    #             # Now convolve with psf
    #             spatial_part = integrated_flux.reshape((self._flat_sky_projection.npix_height,
    #                                                     self._flat_sky_projection.npix_width)).T
    #
    #             # Memorize hoping to reuse it next time
    #             self._spatial_part = spatial_part
    #
    #         # Spectral part
    #         spectral_part = self._spectral_part.get(energy_bin_id, None)
    #
    #         if spectral_part is None:
    #
    #             spectral_part = np.zeros(self._all_fluxes.shape[0])
    #
    #             scale = self._compute_response_scale()
    #
    #             # Loop over the Dec bins that cover this source and compute the expected flux, interpolating between
    #             # two dec bins for each point
    #
    #             for dec_bin1, dec_bin2 in zip(self._dec_bins_to_consider[:-1], self._dec_bins_to_consider[1:]):
    #                 # Get the two response bins to consider
    #                 this_response_bin1 = dec_bin1[energy_bin_id]
    #                 this_response_bin2 = dec_bin2[energy_bin_id]
    #
    #                 # Figure out which pixels are between the centers of the dec bins we are considering
    #                 c1, c2 = this_response_bin1.declination_center, this_response_bin2.declination_center
    #
    #                 idx = (self._flat_sky_projection.decs >= c1) & (self._flat_sky_projection.decs < c2) & \
    #                       self._active_flat_sky_mask
    #
    #                 # Reweight the spectrum separately for the two bins
    #
    #                 # Compute the interpolation weights for the two responses
    #                 w1 = (self._flat_sky_projection.decs[idx] - c2) / (c1 - c2)
    #                 w2 = (self._flat_sky_projection.decs[idx] - c1) / (c2 - c1)
    #
    #                 spectral_part[idx] = (w1 * np.sum(scale * this_response_bin1.sim_signal_events_per_bin) +
    #                                       w2 * np.sum(scale * this_response_bin2.sim_signal_events_per_bin))
    #
    #             # Memorize hoping to reuse it next time
    #             self._spectral_part[energy_bin_id] = spectral_part.reshape((self._flat_sky_projection.npix_height,
    #                                                                         self._flat_sky_projection.npix_width)).T
    #
    #     return self._spectral_part[energy_bin_id] * self._spatial_part
