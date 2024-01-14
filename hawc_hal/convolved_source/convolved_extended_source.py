from __future__ import division

from builtins import zip
from typing import Tuple

import numpy as np
from astromodels import ExtendedSource, Parameter, use_astromodels_memoization
from numba import jit
from numpy.typing import NDArray
from threeML.io.logging import setup_logger

from hawc_hal.flat_sky_projection import FlatSkyProjection
from hawc_hal.response.response import HAWCResponse
from hawc_hal.response.response_bin import ResponseBin

# from ..response.response import HAWCResponse

log = setup_logger(__name__)
log.propagate = False

# Converting from TeV to keV (default units for threeML)
TeVtokeV: float = 1e9

# Conversion factor between deg^2 and rad^2
deg2_to_rad2 = 0.00030461741978670857

ndarray = NDArray[np.float64]
nbdarray = NDArray[np.bool_]


def _select_with_wrap_around(
    coordinate_array: ndarray,
    coord_start: float,
    coord_stop: float,
    wrap: Tuple[float, float] = (360.0, 0.0),
) -> nbdarray:
    """Select pixels covered by an extended source image. The selection is
    based on the range between the coordinted start and the coordinate stopping values

    :param coordinate_array: Array with coordinates defined by the flat
    projection of the ROI
    :param coord_start: Lower bound for the coordinate window
    :type coord_start: float
    :param coord_stop: Upper bound for the coordinate window
    :type coord_stop: float
    :param wrap: Specifies a wrap around to ensure coordinates obey
    the appropriate coordinate boundaries. For RA the wrap around
    is (360, 0) and for declinations it is (-90, 90). Default is (360, 0)
    :type wrap: Tuple[float, float]
    :return: Boolean array that is True only for the pixels covered by
    the spatial image footing
    """

    # NOTE: here arr -> coordinate array, start -> starting coordinate and stop -> stopping coordinate
    # if start > stop:
    #     idx = ((arr >= stop) & (arr <= wrap[0])) | ((arr >= wrap[1]) & (arr <= start))
    # else:
    #     idx = (arr >= start) & (arr <= stop)

    # return idx

    if coord_start > coord_stop:
        return ((coordinate_array >= coord_stop) & (coordinate_array <= wrap[0])) | (
            (coordinate_array >= wrap[1]) & (coordinate_array <= coord_start)
        )

    return (coordinate_array >= coord_start) & (coordinate_array <= coord_stop)


class ConvolvedExtendedSource:
    """Convolve an extended source with the PSF per energy bin"""

    def __init__(
        self,
        source: ExtendedSource,
        response: HAWCResponse,
        flat_sky_projection: FlatSkyProjection,
    ) -> None:
        """Convolve an extended source image with the PSF per energy bin.

        :param source: Extended source object (must instantiated via a likelihood
        model instance)
        :type source: ExtendedSource
        :param response: Detector response object (required to read the response bins)
        :type response: HAWCResponse
        :param flat_sky_projection: Flat sky projection of the ROI (region of interest)
        :type flat_sky_projection: FlatSkyProjection
        """
        # Let's read the essential information for the convolution of
        # of an extended source
        self._response = response
        self._flat_sky_projection = flat_sky_projection
        self._name = source.name
        self._source = source

        # Determine the footprint of the extended source on the flat sky projection
        dec_min, dec_max = self._calculate_foot_print(
            self._flat_sky_projection, self._source
        )

        # Determine the number of declination bins to use for this extended source
        self._dec_bins_to_consider = self._get_dec_bins_to_consider(dec_min, dec_max)

        log.info(
            "Considering %i dec bins for extended source %s"
            % (len(self._dec_bins_to_consider), self._name)
        )

        # Determine the pixels covered by the extended source
        idx_lon = self._get_wrapped_around_ras
        idx_lat = self._get_wrapped_around_decs

        # Select the appropriate pixels covered by the extended source footprint
        self._active_flat_sky_mask = idx_lon & idx_lat

        # Find central bin for the PSF
        self._central_response_bins = self._get_central_response_bins(
            self._source, self._response
        )

        # Get the simulated energy values from the central response bins
        # Since they are defined the same for each response bin, it doesn't
        # matter which one we read them from, so we just read the first available
        # bin.
        self._energy_centers_keV = self.calculate_energy_centers_keV

        self._all_fluxes = np.zeros(
            (self._flat_sky_projection.ras.shape[0], self._energy_centers_keV.shape[0])
        )

    @staticmethod
    def _calculate_foot_print(
        this_flat_sky_projection: FlatSkyProjection, source: ExtendedSource
    ) -> tuple[float, float]:
        """Calculates the footprint of the extended source on a flat sky projection.
        - Maximum declination is determined from minimum value between the declination
        boundaries from extended source and those of the ROI.
        - Minimum declination is determined from maximum value between the boundaries
        in declination from the extended source and those of the ROI.


        :param this_flat_sky_projection: Flat sky projection of the user specified ROI
        :type this_flat_sky_projection: FlatSkyProjection
        :param source: Extended source object (must instantiated via a likelihood model
        instance
        :type source: ExtendedSource
        :return: Declination boundaries necessary to cover the extended source footprint
        :rtype: tuple[float, float]
        """
        # Get the footprint (i.e., the coordinates of the 4 points limiting the
        # projections)
        (
            (_, dec1),
            (_, dec2),
            (_, dec3),
            (_, dec4),
        ) = this_flat_sky_projection.wcs.calc_footprint()  # type: ignore
        # Get the boundaries from the extended source
        _, (lat_start, lat_stop) = source.get_boundaries()

        # Figure out the maximum and minimum declinations to be covered
        dec_min = max(min([dec1, dec2, dec3, dec4]), lat_start)
        dec_max = min(max([dec1, dec2, dec3, dec4]), lat_stop)

        return dec_min, dec_max

    def _get_dec_bins_to_consider(
        self, dec_min: float, dec_max: float
    ) -> list[HAWCResponse]:
        """Calculate the declination bins to consider for the current extended source.

        There is a wrap around to ensure there's always one bin before and after.
        The ROI is assumed to not overlap with the very first or the very last
        dec bin.

        :param dec_min: Lower bound declination value
        :type dec_min: float
        :param dec_max: Upper bound declination value
        :type dec_max: float
        :return: List of HAWC response bins to use for the convolution of the
        extended source
        :rtype: list[HAWCResponse]
        """
        # Get the defined dec bins lower edges, upper edges, and centers
        dec_bins_array = np.array(self._response.dec_bins)
        lower_edges = dec_bins_array[:, 0]
        upper_edges = dec_bins_array[:, -1]
        centers = dec_bins_array[:, 1]

        dec_bins_to_consider_idx = np.flatnonzero(
            (upper_edges >= dec_min) & (lower_edges <= dec_max)
        )
        # Wrap the selection so there's always oen bin before and one after.
        # Note: we assume that the ROI does not overlap with the very first or
        # the very last dec bin
        # Add one dec bin to cover the last part
        dec_bins_to_consider_idx = np.append(
            dec_bins_to_consider_idx, [dec_bins_to_consider_idx[-1] + 1]
        )

        # Add one dec bin to cover the first part
        dec_bins_to_consider_idx = np.insert(
            dec_bins_to_consider_idx, 0, [dec_bins_to_consider_idx[0] - 1]
        )

        dec_bins_to_consider: list[HAWCResponse] = [
            self._response.response_bins[centers[x]] for x in dec_bins_to_consider_idx
        ]
        return dec_bins_to_consider

    @staticmethod
    def _get_central_response_bins(
        source: ExtendedSource, response: HAWCResponse
    ) -> dict[int, ResponseBin]:
        """Retrieve the central response bins for the extended source.

        :param source: An extended source object (must be a likelihood model instance)
        :type source: ExtendedSource
        :param response: HAWC detector response object (required to read the response bins)
        :type response: HAWCResponse
        :return: Dictionary containing the central response bins that coincide with the
        extended source (if interpolation is set to False, then just retrieve the declination
        bin)
        :rtype: dict[int, ResponseBin]
        """
        _, (lat_start, lat_stop) = source.get_boundaries()

        dec_center = (lat_start + lat_stop) / 2.0

        log.info("Central bin is bin at Declination = %.3f" % dec_center)

        central_response_bins = response.get_response_dec_bin(
            dec_center, interpolate=False
        )
        return central_response_bins

    @property
    def _get_wrapped_around_ras(self) -> nbdarray:
        """Find the pixels covered by the right ascension boundaries of the
        extended source. The selection is based on the range between the
        coord start and coord stopping values.

        :return: Returns of boolean array with the right ascension values within the
        extended source boundaries
        :rtype: NDArray[np.bool_]
        """
        (lon_start, lon_stop), _ = self._source.get_boundaries()
        idx_lon = _select_with_wrap_around(
            self._flat_sky_projection.ras, lon_start, lon_stop, (360, 0)
        )
        return idx_lon

    @property
    def _get_wrapped_around_decs(self) -> nbdarray:
        """Find the pixels covered by the declination boundaries of the
        extended source. The selection is based on the range between the
        coord start and coord stopping values.

        :return: Returns of boolean array with the declination values within the
        extended source boundaries
        :rtype: NDArray[np.bool_]
        """
        _, (lat_start, lat_stop) = self._source.get_boundaries()
        idx_lat = _select_with_wrap_around(
            self._flat_sky_projection.decs, lat_start, lat_stop, (90, -90)
        )
        return idx_lat

    @property
    def calculate_energy_centers_keV(self) -> ndarray:
        """Retrieve the energy centers from the central response bins.

        :return:  Returns the energy centers in units of TeV (converted from keV which
        is the convention used in threeML)
        :rtype: NDArray[np.float64]
        """
        return (
            self._central_response_bins[
                list(self._central_response_bins.keys())[0]
            ].sim_energy_bin_centers
            * TeVtokeV
        )

    def _setup_callbacks(self, callback) -> None:
        """Register the callback for parameter changes that are free and those that
        are linked. Check if the parameter has changed, if so then register a callback.
        This is put in place to prevent unnecessary flux calculations for extended sources.

        :param callback: Callcack function to register
        :type callback:
        """
        # Register call back with all free parameters and all linked parameters.
        # If a parameter is linked to another one, the other one might be in
        # a different source, so we add a callback to that one so that we
        # recompute this source when needed.
        for parameter in list(self._source.parameters.values()):
            if parameter.free:
                parameter.add_callback(callback)

            if parameter.has_auxiliary_variable:
                # Add a callback to the auxiliary variable so that when
                #  that is changed, we need to recompute the model
                aux_variable, _ = parameter.auxiliary_variable

                aux_variable.add_callback(callback)

    def get_source_map(self, energy_bin_id: str, tag=None):
        """Register a callback for changes in the free parameters of the extended
        source. It keeps tracks of when a parameter changes. When a change occurs the
        function will register a callback which will indicate that the source
        flux need to be computed inside the get_source_map() method. This
        placed so that the computation is avoided when no parameters changes
        during sampling.

        :param this_parameter: Astromodels parameer instance capable of registering
        callbacks for when its value changes.
        :type this_parameter: Parameter
        """
        raise NotImplementedError("You need to implement this in derived classes")


class ConvolvedExtendedSource3D(ConvolvedExtendedSource):
    """A class used to represent a 3D convolved extended source.

    This class is a child of the ConvolvedExtendedSource class and
    is used to handle 3D convolved extended sources.

    """

    def __init__(
        self,
        source: ExtendedSource,
        response: HAWCResponse,
        flat_sky_projection: FlatSkyProjection,
    ):
        """Convolved extended source class

        :param source: Extended source instance defined in the likelihood model
        :param response: HAWC detector response object (required to read the response bins)
        :param flat_sky_projection: Flat sky projection of the ROI
        """

        super().__init__(source, response, flat_sky_projection)

        # We implement a caching system so that the source
        # flux is evaluated only when strictly needed,
        # because it is the most computationally intense part otherwise.
        self._recompute_flux = True

        # Register callback to keep track of the parameter changes
        self._setup_callbacks(self._parameter_change_callback)

    def _parameter_change_callback(self, this_parameter: Parameter) -> None:
        """Register a callback for changes in the free parameters of the extended
        source. It keeps tracks of when a parameter changes. When a change occurs the
        function will register a callback which will indicate that the source
        flux need to be computed inside the get_source_map() method. This
        placed so that the computation is avoided when no parameters changes
        during sampling.

        :param this_parameter: Astromodels parameer instance capable of registering
        callbacks for when its value changes.
        :type this_parameter: Parameter
        """
        # A parameter has changed, we need to recompute the function.
        # NOTE: we do not recompute it here because if more than one
        # parameter changes at the time (like for example during sampling) we
        # do not want to recompute the function multiple time before we get
        # to the convolution stage. Therefore we will compute the function
        # in the get_source_map method

        # log.debug("%s has changed" % this_parameter.name)
        self._recompute_flux = True

    @staticmethod
    @jit(nopython=True, parallel=False)
    def _calculate_scale(
        fluxes_array: ndarray,
        response_bin_sim_diff_photon_fluxes: ndarray,
        pixel_area_rad2: float,
    ) -> ndarray:
        """Reweights the spectrum separately for two response bins.

        :param fluxes_array:  Array of fluxes
        :type fluxes_array: NDArray[np.float64]
        :param response_bin_sim_diff_photon_fluxes: Differential photon fluxes per bin
        :type response_bin_sim_diff_photon_fluxes: NDArray[np.float64]
        :param pixel_area_rad2: Pixel area in units of rad^2
        :type fluxes_array: NDArray[np.float64]
        :return: Returns the scale of fluxes array from the model expected simulated
        differential fluxes from the response bins. The spectrum is weighted separately
        for two response bins. The scale is the same because the sim_differential_photon_fluxes
        are the same (the simulation used to make the response use the same spectrum for
        each bin). What changes between the two bins is the observed signal per bin (the
        sim_signal_events_per_bin member).
        :rtype: NDArray[np.float64]
        """
        # NOTE: the sim_differential_photon_fluxes are the same for every bin,
        # so it doesn't matter which they are read from
        return np.divide(
            (fluxes_array * pixel_area_rad2),
            response_bin_sim_diff_photon_fluxes,
        )

    @staticmethod
    def _calculate_interpolation_weights(
        flat_sky_projection_decs_array: ndarray,
        response_bin1_center: ndarray,
        response_bin2_center: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Calculate the interpolation weights between two central response bins that
        are within the footprint of the extended source.

        :param flat_sky_projection_decs_array: Array of declination coordinates
        on the flat sky projection
        :type flat_sky_projection_decs_array: NDArray[np.float64]
        :param response_bin1_center: First central response bin
        :type response_bin1_center: NDArray[np.float64]
        :param response_bin2_center: Second central response bin
        :type response_bin2_center: NDArray[np.float64]
        :return: the interpolation weights for the two responses considered
        :rtype: tuple[NDArray[np.float64], NDArray[np.float64]]
        """
        # Compute the interpolation weights for the two responses
        w1 = np.divide(
            (flat_sky_projection_decs_array - response_bin2_center),
            (response_bin1_center - response_bin2_center),
        )
        w2 = np.divide(
            (flat_sky_projection_decs_array - response_bin1_center),
            (response_bin2_center - response_bin1_center),
        )

        return w1, w2

    @staticmethod
    def _calculate_model_image(
        weight1: ndarray,
        weight2: ndarray,
        scale: ndarray,
        response_bin1_signal_evts_per_bin: ndarray,
        response_bin2_signal_evts_per_bin: ndarray,
    ) -> ndarray:
        """Evaluate the model image.

        :param weight1: Computed weight for the first response bin
        :type weight1: NDArray[np.float64]
        :param weight2: Computed weight for the second response bin
        :type weight1: NDArray[np.float64]
        :param scale: Scale of fluxes array from the model expected simulated
        :type scale: NDArray[np.float64]
        :param response_bin1_signal_evts_per_bin: Number of signal events for first bin
        considered
        :type weight1: NDArray[np.float64]
        :param response_bin2_signal_evts_per_bin: Number of signal events for the second bin
        considered
        :type weight1: NDArray[np.float64]
        :return: Returns model image
        :rtype: NDArray[np.float64]
        """
        return (
            weight1
            * np.sum(
                scale * response_bin1_signal_evts_per_bin,
                axis=1,
            )
            + weight2
            * np.sum(
                scale * response_bin2_signal_evts_per_bin,
                axis=1,
            )
        ) * TeVtokeV

    def get_source_map(
        self,
        energy_bin_id: str,
        tag=None,
    ) -> ndarray:
        """Perform the convolution of an extended source map with the PSF per energy bin.

        :param tag: Optional tag to ensure sources are treated uniquely
        :type tag: str
        :param energy_bin_id: Analysis bin id defined in HAWC's maptree and response files
        :type energy_bin_id: str
        :return: Convolved image only for the pixels within the flat sky projection
        :rtype: NDArray[np.float64]
        """

        # We do not use the memoization in astromodels because we are doing
        # caching by ourselves, so the astromodels memoization would turn
        # into 100% cache miss and use a lot of RAM for nothing,
        # given that we are evaluating the function on many points and many energies

        with use_astromodels_memoization(False):
            # If we need to recompute the flux, let's do it
            if self._recompute_flux:
                # print("recomputing %s" % self._name)

                # Recompute the fluxes for the pixels that are covered by this extended source
                self._all_fluxes[self._active_flat_sky_mask, :] = self._source(
                    self._flat_sky_projection.ras[self._active_flat_sky_mask],
                    self._flat_sky_projection.decs[self._active_flat_sky_mask],
                    self._energy_centers_keV,
                )  # 1 / (keV cm^2 s rad^2)

                # We don't need to recompute the function anymore until a parameter changes
                self._recompute_flux = False

            # Now compute the expected signal
            pixel_area_rad2 = (
                self._flat_sky_projection.project_plane_pixel_area * deg2_to_rad2
            )

            this_model_image = np.zeros(self._all_fluxes.shape[0])

            # Loop over the Dec bins that cover this source and compute the expected flux, interpolating between
            # two dec bins for each point
            for dec_bin1, dec_bin2 in zip(
                self._dec_bins_to_consider[:-1], self._dec_bins_to_consider[1:]
            ):
                # Get the two response bins to consider
                this_response_bin1: ResponseBin = dec_bin1[energy_bin_id]
                this_response_bin2: ResponseBin = dec_bin2[energy_bin_id]

                # Figure out which pixels are between the centers of the dec bins we are considering
                c1, c2 = (
                    this_response_bin1.declination_center,
                    this_response_bin2.declination_center,
                )

                idx = (
                    (self._flat_sky_projection.decs >= c1)
                    & (self._flat_sky_projection.decs < c2)
                    & self._active_flat_sky_mask
                )

                # Reweight the spectrum separately for the two bins

                # NOTE: the scale is the same because the sim_differential_photon_fluxes are the same (the simulation
                # used to make the response used the same spectrum for each bin). What changes between the two bins
                # is the observed signal per bin (the .sim_signal_events_per_bin member)
                scale = self._calculate_scale(
                    self._all_fluxes[idx, :],
                    this_response_bin1.sim_differential_photon_fluxes,
                    pixel_area_rad2,
                )

                # Compute the interpolation weights for the two responses
                w1, w2 = self._calculate_interpolation_weights(
                    self._flat_sky_projection.decs[idx], c1, c2
                )

                # Compute the model image
                this_model_image[idx] = self._calculate_model_image(
                    w1,
                    w2,
                    scale,
                    this_response_bin1.sim_signal_events_per_bin,
                    this_response_bin2.sim_signal_events_per_bin,
                )

            # Reshape the flux array into an image
            this_model_image = this_model_image.reshape(
                (
                    self._flat_sky_projection.npix_height,
                    self._flat_sky_projection.npix_width,
                )
            ).T

            return this_model_image


class ConvolvedExtendedSource2D(ConvolvedExtendedSource3D):
    """Class for a 2D convolved extended source. This class inherits all the methods
    from the more complex 3D extended source class, but it is simpler to use
    and much faster to compute.

    """

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
