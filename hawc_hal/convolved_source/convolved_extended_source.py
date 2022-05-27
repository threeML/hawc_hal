from __future__ import division
from builtins import zip
from past.utils import old_div
from builtins import object
import numpy as np

from astromodels import use_astromodels_memoization
from threeML.io.logging import setup_logger
log = setup_logger(__name__)
log.propagate = False


def _select_with_wrap_around(arr, start, stop, wrap=(360, 0)):

    if start > stop:

        # This can happen if for instance lon_start = 280 and lon_stop = 10, which means we
        # should keep between 280 and 360 and then between 0 to 10
        idx = ((arr >= stop) & (arr <= wrap[0])) | ((arr >= wrap[1]) & (arr <= start))

    else:

        idx = (arr >= start) & (arr <= stop)

    return idx

# Conversion factor between deg^2 and rad^2
deg2_to_rad2 = 0.00030461741978670857


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
        lower_edges = np.array([x[0] for x in response.dec_bins])
        upper_edges = np.array([x[-1] for x in response.dec_bins])
        centers = np.array([x[1] for x in response.dec_bins])

        dec_bins_to_consider_idx = np.flatnonzero((upper_edges >= dec_min) & (lower_edges <= dec_max))

        # Wrap the selection so we have always one bin before and one after.
        # NOTE: we assume that the ROI do not overlap with the very first or the very last dec bin
        # Add one dec bin to cover the last part
        dec_bins_to_consider_idx = np.append(dec_bins_to_consider_idx, [dec_bins_to_consider_idx[-1] + 1])
        # Add one dec bin to cover the first part
        dec_bins_to_consider_idx = np.insert(dec_bins_to_consider_idx, 0, [dec_bins_to_consider_idx[0] - 1])

        self._dec_bins_to_consider = [response.response_bins[centers[x]] for x in dec_bins_to_consider_idx]

        log.info("Considering %i dec bins for extended source %s" % (len(self._dec_bins_to_consider),
                                                                  self._name))

        # Find central bin for the PSF

        dec_center = (lat_start + lat_stop) / 2.0
        #
        self._central_response_bins = response.get_response_dec_bin(dec_center, interpolate=False)

        log.info("Central bin is bin at Declination = %.3f" % dec_center)

        # Take note of the pixels within the flat sky projection that actually need to be computed. If the extended
        # source is significantly smaller than the flat sky projection, this gains a substantial amount of time

        idx_lon = _select_with_wrap_around(self._flat_sky_projection.ras, lon_start, lon_stop, (360, 0))
        idx_lat = _select_with_wrap_around(self._flat_sky_projection.decs, lat_start, lat_stop, (90, -90))

        self._active_flat_sky_mask = (idx_lon & idx_lat)

        assert np.sum(self._active_flat_sky_mask) > 0, "Mismatch between source %s and ROI" % self._name

        # Get the energies needed for the computation of the flux
        self._energy_centers_keV = self._central_response_bins[list(self._central_response_bins.keys())[0]].sim_energy_bin_centers * 1e9

        # Prepare array for fluxes
        self._all_fluxes = np.zeros((self._flat_sky_projection.ras.shape[0],
                                     self._energy_centers_keV.shape[0]))

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
                scale = old_div((self._all_fluxes[idx, :] * pixel_area_rad2), this_response_bin1.sim_differential_photon_fluxes)

                # Compute the interpolation weights for the two responses
                w1 = old_div((self._flat_sky_projection.decs[idx] - c2), (c1 - c2))
                w2 = old_div((self._flat_sky_projection.decs[idx] - c1), (c2 - c1))

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
