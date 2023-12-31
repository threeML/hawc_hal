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
    """
    Select pixels covered by an extended source image. The selection is based on
    the range between the coord start and coord stopping values.

    Parameters
    ----------
    coordinate_array : np.ndarray
        Array holding coordinates defined by the flat projecting of the ROI
    coord_start : float
        Lower bound range coordinate value
    coord_stop : float
        Upper bound range coordinate value
    wrap : Tuple[float, float], optional
        wrap around for the coordinate values for RA: (360, 0) and
        for declinations it is (-90, 90), by default (360.0, 0.0)

    Returns
    -------
    NDArray[np.bool_]
        returns a boolean array with the same shape as the coordinate array

    Notes:
    ------
    The wrap around is in place to ensure coordinates obey the boundaries correctly.
    For RA, the boundary is set at (360, 0). For declinations, the boundary is set
    at (-90, 90).
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
        """Convolve an extended source with the PSF per energy bin.

        Parameters
        ----------
        source : ExtendedSource
            An extended source object (must be a likelihood model instance)
        response : HAWCResponse
            A HAWC detector response object (required to read the response bins)
        flat_sky_projection : FlatSkyProjection
            Flat sky projection of the ROI (region of interest)
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
        """Calculates the footprint of the extended source on the flat sky projection.

        Parameters
        ----------
        this_flat_sky_projection : FlatSkyProjection
            Flat sky projection of the ROI
        source : ExtendedSource
            An extended source object (must be a likelihood model instance)

        Returns
        -------
        Tuple[float, float]
            The minimum and maximum declination values to be covered by the extended source
        """
        # Get the footprint (i.e., the coordinates of the 4 points limiting the projections)
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
        """Calculate the declination bins to consider for the extended source.

        Parameters
        ----------
        dec_min : float
            Lower bound declination value
        dec_max : float
            Upper bound declination value

        Returns
        -------
        list[HAWCResponse]
            Ordered list of HAWC response bins to consider for the extended source

        Notes
        ------
            There is a wrap around to ensure there's always one bin before and after.
            The ROI is assumed to not overlap with the very first or the very last
            dec bin.
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

        Parameters
        ----------
        source : ExtendedSource
            An extended source object (must be a likelihood model instance)
        response : HAWCResponse
            HAWC detector response object (required to read the response bins)

        Returns
        -------
        dict[int, ResponseBin]
            Dictionary containing the central response bins for the extended source
            (if interpolation is set to False, then just retrieve the declination bin)
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
        """
        Find the pixels covered by the right ascension boundaries of the
        extended source. The selection is based on the range between the
        coord start and coord stopping values.

        Returns
        -------
        NDArray[np.bool_]
            Returns of boolean array with the right ascension values within the
            extended source boundaries
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

        Returns
        -------
        NDArray[np.bool_]
            Returns of boolean array with the declination values within the
            extended source boundaries
        """
        _, (lat_start, lat_stop) = self._source.get_boundaries()
        idx_lat = _select_with_wrap_around(
            self._flat_sky_projection.decs, lat_start, lat_stop, (90, -90)
        )
        return idx_lat

    @property
    def calculate_energy_centers_keV(self) -> ndarray:
        """Retrieve the energy centers from the central response bins.

        Returns
        -------
        NDArray[np.float64]
            Returns the energy centers in units of keV (as required by threeML)
        Notes
        -----
        The definition of energy centers is the same for all bins, so it doesn't matter which bin we read them from. The energy centers are converted to keV (as required by threeML).
        """
        return (
            self._central_response_bins[
                list(self._central_response_bins.keys())[0]
            ].sim_energy_bin_centers
            * TeVtokeV
        )

    def _setup_callbacks(self, callback) -> None:
        """Register a callback for changes in all free parameters and all linked parameters. Check if a parameter is link to another, the other might be
        connected to a different source, so we add a callback to that so that the flux
        is recomputed if needed.

        Parameters
        ----------
        callback : function
            Callback to parameters that have changed values
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
        """Perform the convolution of an extended source map with the PSF per energy bin.

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin id defined in HAWC's maptree and response files
        tag : str, optional
            Optional tag to define one source from the other, by default None

        Returns
        -------
        NDAarray[np.float64]
            Convolved imaged only for the pixels within the flat sky projection
        """
        raise NotImplementedError("You need to implement this in derived classes")


class ConvolvedExtendedSource3D(ConvolvedExtendedSource):
    """
    A class used to represent a 3D convolved extended source.

    This class is a child of the ConvolvedExtendedSource class and
    is used to handle 3D convolved extended sources.

    """

    def __init__(
        self,
        source: ExtendedSource,
        response: HAWCResponse,
        flat_sky_projection: FlatSkyProjection,
    ):
        """Convolved extended source

        Parameters
        ----------
        source : ExtendedSource
            Extended source object (must be likelihood model instance)
        response : HAWCResponse
            HAWC detector response object (required to read the response bins)
        flat_sky_projection : FlatSkyProjection
            Flat sky projection of the ROI
        """

        super().__init__(source, response, flat_sky_projection)

        # We implement a caching system so that the source
        # flux is evaluated only when strictly needed,
        # because it is the most computationally intense part otherwise.
        self._recompute_flux = True

        # Register callback to keep track of the parameter changes
        self._setup_callbacks(self._parameter_change_callback)

    def _parameter_change_callback(self, this_parameter: Parameter) -> None:
        """Register a callback for changes in the free parameters of the extended sources.

        Parameters
        ----------
        this_parameter : Parameter
            An astromodels parameter instance capable of registering callbacks for
            when parameters change values.
        Notes
        -----
        This function is called when a parameter changes value. It is used to keep track of the parameter values. When there is a parameter change the function will recompute the flux. The flux is not recomputed immediately, but only when the get_source_map method is called. This is done to avoid recomputing the flux multiple times when more than one parameter changes at the same time (like for example during sampling).
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
        """
        Reweights the spectrum separately for two response bins

        Parameters
        ----------
        fluxes_array : NDArray[np.float64]
            Array of fluxes
        response_bin_sim_diff_photon_fluxes : NDArray[np.float64]
            Differential photon fluxes per bin
        pixel_area_rad2 : float
            Pixel area in units of rad^2

        Returns
        -------
        NDArray[np.float64]
            Returns the scale of fluxes array from the model expected simulated
            differential fluxes from the response bins.
        Notes
        -----
        The spectrum is weighted separately for two response bins. The scale is the
        same because the sim_differential_photon_fluxes are the same (the simulation
        used to make the response use the same spectrum for each bin). What changes
        between the two bins is the observed signal per bin (the
        sim_signal_events_per_bin member).
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
        """Calculate the interpolation weights for the two responses.

        Parameters
        ----------
        flat_sky_projection_decs_array : NDArray[np.float64]
            Array of declinations
        response_bin1_center : NDArray[np.float64]
            Response bin object
        response_bin2_center : NDArray[np.float64]
            Response bin object

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            Returns the interpolation weights for the two responses
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
        """Calculate the model image.

        Parameters
        ----------
        weight1 : NDArray[np.float64]
            Array of weights
        weight2 : NDArray[np.float64]
            Array of weights
        scale : NDArray[np.float64]
            Array of scales
        response_bin1_signal_evts_per_bin : NDArray[np.float64]
            Response bin object
        response_bin2_signal_evts_per_bin : NDArray[np.float64]
            Response bin object

        Returns
        -------
        NDArray[np.float64]
            Returns the model image
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

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin id defined in HAWC's maptree and response files
        tag : str, optional
            Optional tag to define one source from the other, by default None

        Returns
        -------
        NDAarray[np.float64]
            Convolved imaged only for the pixels within the flat sky projection
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
    """Class for convolved 2D extended sources. It inherits all the methods
    from the more complex 3D extended source class, but it is simpler to use
    and much faster to compute.

    Parameters
    ----------
    ConvolvedExtendedSource3D : ConvolvedExtendedSource
        Convolved extended source class
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
