from __future__ import division

import collections
import contextlib
import copy
from builtins import range, str
from threading import RLock
from typing import Callable, Optional

import astromodels
import healpy as hp
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from astromodels import Parameter
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft as convolve
from numpy.typing import NDArray
from past.utils import old_div
from scipy.stats import poisson
from threeML.io.logging import setup_logger
from threeML.parallel import parallel_client
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.gammaln import logfactorial
from tqdm.auto import tqdm

from hawc_hal.convolved_source import (
    ConvolvedExtendedSource2D,
    ConvolvedExtendedSource3D,
    ConvolvedPointSource,
    ConvolvedSourcesContainer,
)
from hawc_hal.flat_sky_projection import FlatSkyProjection
from hawc_hal.healpix_handling import (
    FlatSkyToHealpixTransform,
    SparseHealpix,
    get_gnomonic_projection,
)
from hawc_hal.log_likelihood import log_likelihood
from hawc_hal.maptree import map_tree_factory
from hawc_hal.maptree.data_analysis_bin import DataAnalysisBin
from hawc_hal.maptree.map_tree import MapTree
from hawc_hal.psf_fast import PSFConvolutor
from hawc_hal.region_of_interest.healpix_cone_roi import HealpixConeROI
from hawc_hal.region_of_interest.healpix_map_roi import HealpixMapROI
from hawc_hal.response import hawc_response_factory
from hawc_hal.util import ra_to_longitude

log = setup_logger(__name__)
log.propagate = False

ndarray = NDArray[np.float64]


class HAL(PluginPrototype):
    """The HAWC Accelerated Likelihood plugin for 3ML."""

    # """
    # The HAWC Accelerated Likelihood plugin for 3ML.
    # :param name: name for the plugin
    # :param maptree: Map Tree (either ROOT or hdf5 format)
    # :param response: Response of HAWC (either ROOT or hd5 format)
    # :param roi: a ROI instance describing the Region Of Interest
    # :param flat_sky_pixels_size: size of the pixel for the flat sky projection (Hammer Aitoff)
    # """

    def __init__(
        self,
        name: str,
        maptree: str,
        response_file: str,
        roi: HealpixConeROI | HealpixMapROI,
        flat_sky_pixels_size: float = 0.17,
        n_workers: int = 1,
        set_transits: Optional[float] = None,
    ):
        """The HAWC Accelerated Likelihood plugin for 3ML.

        Parameters
        ----------
        name : str
            HAL instance name
        maptree : str
            HAWC map tree file (either ROOT or hdf5 format)
        response_file : str
            HAWC response file (either ROOT or hdf5 format)
        roi : HealpixConeROI | HealpixMapROI
            Region of interest (ROI)
        flat_sky_pixels_size : float, optional
            Size of pixel for the flat sky projection (Hammer Aitoff), by default 0.17
        n_workers : int, optional
            Number of workers used for multiprocessing (experimental for now), by default 1
        set_transits : Optional[float], optional
            Specify the number of transits if not using the maximum number read from the map tree, by default None
        """
        # Store ROI
        self._roi = roi
        self._n_workers = n_workers
        self.lock = RLock()

        # optionally specify n_transits
        if set_transits is not None:
            log.info(f"Setting transits to {set_transits}")
            n_transits = set_transits

        else:
            n_transits = None
            log.info("Using maximum number of transits from maptree...")

        # Set up the flat-sky projection
        self.flat_sky_pixels_size = flat_sky_pixels_size
        self._flat_sky_projection: FlatSkyProjection = (
            self._roi.get_flat_sky_projection(self.flat_sky_pixels_size)
        )

        # Read map tree (data)
        self._maptree = map_tree_factory(
            maptree, roi=self._roi, n_transits=n_transits, n_workers=self._n_workers
        )

        # Read detector response_file
        self._response = hawc_response_factory(
            response_file_name=response_file, n_workers=self._n_workers
        )

        # Get dictionaries for caching
        # TODO: figure out a smart way of caching
        # self._cached_pnt_source_params = CacheDict(size_limit=20, n_elements_kept=10)

        # Use a renormalization of the background as nuisance parameter
        # NOTE: it is fixed to 1.0 unless the user explicitly sets it free (experimental)
        self._nuisance_parameters = collections.OrderedDict()
        # self._nuisance_parameters['%s_bkg_renorm' % name] = Parameter('%s_bkg_renorm' % name, 1.0,
        self._nuisance_parameters[f"{name}_bkg_renorm"] = Parameter(
            f"{name}_bkg_renorm",
            1.0,
            min_value=0.5,
            max_value=1.5,
            delta=0.01,
            desc="Renormalization for background map",
            free=False,
            is_normalization=False,
        )

        # Instance parent class

        # super(HAL, self).__init__(name, self._nuisance_parameters)
        # python3 new way of doing things
        super().__init__(name, self._nuisance_parameters)

        self._likelihood_model: Optional[astromodels.Model | None] = None

        # These lists will contain the maps for the point sources
        self._convolved_point_sources = ConvolvedSourcesContainer()
        # and this one for extended sources
        self._convolved_ext_sources = ConvolvedSourcesContainer()

        # All energy/nHit bins are loaded in memory
        self._all_planes: list[str] = list(self._maptree.analysis_bins_labels)

        # The active planes list always contains the list of *indexes* of the active planes
        self._active_planes: Optional[list[str] | None] = None

        # Set up the transformations from the flat-sky projection to Healpix, as well as the list of active pixels
        # (one for each energy/nHit bin). We make a separate transformation because different energy bins might have
        # different nsides
        self._active_pixels = collections.OrderedDict()
        self._flat_sky_to_healpix_transform = collections.OrderedDict()

        for bin_id in self._maptree:
            this_maptree = self._maptree[bin_id]
            this_nside = this_maptree.nside
            this_active_pixels = roi.active_pixels(this_nside)

            this_flat_sky_to_hpx_transform = FlatSkyToHealpixTransform(
                self._flat_sky_projection.wcs,
                "icrs",
                this_nside,
                this_active_pixels,
                (
                    self._flat_sky_projection.npix_width,
                    self._flat_sky_projection.npix_height,
                ),
                order="bilinear",
            )

            self._active_pixels[bin_id] = this_active_pixels
            self._flat_sky_to_healpix_transform[bin_id] = this_flat_sky_to_hpx_transform

        # This will contain a list of PSF convolutors for extended sources, if there is any in the model

        self._psf_convolutors: Optional[dict[str, PSFConvolutor] | None] = None

        # Pre-compute the log-factorial factor in the likelihood, so we do not keep to computing it over and over
        # again.
        self._log_factorials = collections.OrderedDict()

        # We also apply a bias so that the numerical value of the log-likelihood stays small. This helps when
        # fitting with algorithms like MINUIT because the convergence criterium involves the difference between
        # two likelihood values, which would be affected by numerical precision errors if the two values are
        # too large
        self._saturated_model_like_per_maptree = collections.OrderedDict()

        # The actual computation is in a method so we can recall it on clone (see the get_simulated_dataset method)
        self._compute_likelihood_biases()

        # This will save a clone of self for simulations
        self._clone = None

        # Integration method for the PSF (see psf_integration_method)
        self._psf_integration_method: str = "exact"

    def __getstate__(self):
        """Remove the lock before pickling"""
        state = self.__dict__.copy()
        del state["lock"]
        return state

    def __setstate__(self, state) -> None:
        """Set the lock after unpickling"""
        self.__dict__.update(state)
        self.lock = RLock()

    @property
    def psf_integration_method(self) -> str:
        """
        Get or set the method for the integration of the PSF.

        Parameters
        ----------

            * "exact" is more accurate but slow, if the position is free to vary it adds a lot of time to the fit. This is
            the default, to be used when the position of point sources are fixed. The computation in that case happens only
            once so the impact on the run time is negligible.
            * "fast" is less accurate (up to an error of few percent in flux) but a lot faster. This should be used when
            the position of the point source is free, because in that case the integration of the PSF happens every time
            the position changes, so several times during the fit.

        Notes:
        ------
            If you have a fit with a free position, use "fast". When the position is found, you can fix it, switch to
            "exact" and redo the fit to obtain the most accurate measurement of the flux. For normal sources the difference
            will be small, but for very bright sources it might be up to a few percent (most of the time < 1%). If you are
            interested in the localization contour there is no need to rerun with "exact". Either "exact" or "fast"

        """

        return self._psf_integration_method

    @psf_integration_method.setter
    def psf_integration_method(self, mode):
        assert mode.lower() in [
            "exact",
            "fast",
        ], "PSF integration method must be either 'exact' or 'fast'"

        self._psf_integration_method = mode.lower()

    def _setup_psf_convolutors(self, source_declination: float):
        """Set up the PSF convolutors at the source declination

        Parameters
        ----------
        source_declination : float
            Source declination in degrees
        """
        # central_response_bins = self._response.get_response_dec_bin(
        #     self._roi.ra_dec_center[1]
        # )
        central_response_bins = self._response.get_response_dec_bin(source_declination)

        self._psf_convolutors: dict[str, PSFConvolutor] = collections.OrderedDict()
        for bin_id in central_response_bins:
            # Only set up PSF convolutors for active bins.
            if bin_id in self._active_planes:
                self._psf_convolutors[bin_id] = PSFConvolutor(
                    central_response_bins[bin_id].psf, self._flat_sky_projection
                )

    def _compute_likelihood_biases(self):
        for bin_label in self._maptree:
            data_analysis_bin = self._maptree[bin_label]

            this_log_factorial = np.sum(
                logfactorial(data_analysis_bin.observation_map.as_partial().astype(int))
            )
            self._log_factorials[bin_label] = this_log_factorial

            # As bias we use the likelihood value for the saturated model
            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            sat_model = np.clip(obs - bkg, 1e-50, None).astype(np.float64)

            self._saturated_model_like_per_maptree[bin_label] = (
                log_likelihood(obs, bkg, sat_model) - this_log_factorial
            )

    def get_saturated_model_likelihood(self):
        """
        Returns the likelihood for the saturated model (i.e. a model exactly equal to observation - background).

        :return:
        """
        return sum(self._saturated_model_like_per_maptree.values())

    def set_active_measurements(
        self, bin_id_min=None, bin_id_max=None, bin_list=None
    ) -> None:
        """
        Set the active analysis bins to use during the analysis. It can be used in two ways:

        - Specifying a range: if the response and the maptree allows it, you can specify a minimum id and a maximum id
        number. This only works if the analysis bins are numerical, like in the normal fHit analysis. For example:

            > set_active_measurement(bin_id_min=1, bin_id_max=9)

        - Specifying a list of bins as strings. This is more powerful, as allows to select any bins, even
        non-contiguous bins. For example:

            > set_active_measurement(bin_list=[list])

        :param bin_id_min: minimum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_id_max: maximum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_list: a list of analysis bins to use
        :return: None
        """

        # Check for legal input
        # ? LEGACY CODE: naming scheme of bins has changed and no longer follows
        # ? the integer nHit convention. Should this code be removed?
        if bin_id_min is not None:
            assert (
                bin_id_max is not None
            ), "If you provide a minimum bin, you also need to provide a maximum bin."

            # Make sure they are integers
            bin_id_min = int(bin_id_min)
            bin_id_max = int(bin_id_max)

            self._active_planes = []
            for this_bin in range(bin_id_min, bin_id_max + 1):
                this_bin = str(this_bin)
                if this_bin not in self._all_planes:
                    raise ValueError(
                        f"Bin {this_bin} is not contained in this maptree."
                    )

                self._active_planes.append(this_bin)

        else:
            assert (
                bin_id_max is None
            ), "If you provie a maximum bin, you also need to provide a minimum bin."

            assert bin_list is not None

            self._active_planes = []

            for this_bin in bin_list:
                if this_bin not in self._all_planes:
                    raise ValueError(
                        f"Bin {this_bin} is not contained in this maptree."
                    )

                self._active_planes.append(this_bin)

        if self._likelihood_model:
            self.set_model(self._likelihood_model)

    def display(self, verbose=False) -> None:
        """
        Prints summary of the current object content.
        """

        log.info("Region of Interest: ")
        log.info("-------------------")
        self._roi.display()

        log.info("")
        log.info("Flat sky projection: ")
        log.info("--------------------")

        log.info(
            f"Width x height {self._flat_sky_projection.npix_width} x {self._flat_sky_projection.npix_height} px"
        )
        # log.info("Width x height: %s x %s px" % (self._flat_sky_projection.npix_width,
        #                                      self._flat_sky_projection.npix_height))
        log.info(f"Pixel sizes: {self._flat_sky_projection.pixel_size} deg")
        # log.info("Pixel sizes: %s deg" % self._flat_sky_projection.pixel_size)

        log.info("")
        log.info("Response: ")
        log.info("---------")

        self._response.display(verbose)

        log.info("")
        log.info("Map Tree: ")
        log.info("----------")

        self._maptree.display()

        log.info("")
        # log.info("Active energy/nHit planes ({}):".format(len(self._active_planes)))
        log.info(f"Active energy/nHit planes ({len(self._active_planes)}):")
        log.info("-------------------------------")
        log.info(self._active_planes)

    def set_model(self, likelihood_model_instance: astromodels.Model) -> None:
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._likelihood_model = likelihood_model_instance

        # Reset
        self._convolved_point_sources.reset()
        self._convolved_ext_sources.reset()

        # For each point source in the model, build the convolution class
        pnt_sources: list[astromodels.PointSource] = list(
            self._likelihood_model.point_sources.values()
        )

        if pnt_sources:
            for source in pnt_sources:
                this_convolved_point_source = ConvolvedPointSource(
                    source, self._response, self._flat_sky_projection
                )

                self._convolved_point_sources.append(this_convolved_point_source)

        # Samewise for extended sources
        ext_sources: list[astromodels.ExtendedSource] = list(
            self._likelihood_model.extended_sources.values()
        )

        # NOTE: ext_sources evaluate to False if empty
        if ext_sources:
            # We will need to convolve
            roi_dec_center: float = self._roi.ra_dec_center[1]

            for source in ext_sources:
                #  Choosing the center location of the PSF at center of ROI
                #  may pose a significant issue for very large ROIs
                if hasattr(source.spatial_shape, "lat0"):
                    # load normal extended source model
                    self._setup_psf_convolutors(source.spatial_shape.lat0.value)

                else:
                    # loading a template source model if no lat0 available from
                    # spatial_shape
                    self._setup_psf_convolutors(roi_dec_center)

                if source.spatial_shape.n_dim == 2:
                    this_convolved_ext_source = ConvolvedExtendedSource2D(
                        source, self._response, self._flat_sky_projection
                    )

                else:
                    this_convolved_ext_source = ConvolvedExtendedSource3D(
                        source, self._response, self._flat_sky_projection
                    )

                self._convolved_ext_sources.append(this_convolved_ext_source)

    def get_excess_background(self, ra: float, dec: float, radius: float):
        """Calculates excess (data-bkg), background, and model counts at
        different radial distances from origin of radial profile.


        Parameters
        ----------
        ra : float
            RA of origin of radial profile
        dec : float
           Dec of origin of radial profile
        radius : float
           distance from origin of radial profile

        Returns
        -------
           returns a tuple of numpy arrays with info of areas (steradian) and
           signal excess, background, and model in units of counts to be used
           in the get_radial_profile method.
        """

        radius_radians = np.deg2rad(radius)

        total_counts = np.zeros(len(self._active_planes), dtype=float)
        background = np.zeros_like(total_counts)
        observation = np.zeros_like(total_counts)
        model = np.zeros_like(total_counts)
        signal = np.zeros_like(total_counts)
        area = np.zeros_like(total_counts)

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        longitude = ra_to_longitude(ra)
        latitude = dec
        center = hp.ang2vec(longitude, latitude, lonlat=True)

        for i, energy_id in enumerate(self._active_planes):
            data_analysis_bin: DataAnalysisBin = self._maptree[energy_id]
            this_nside = data_analysis_bin.observation_map.nside

            radial_bin_pixels = hp.query_disc(
                this_nside,
                center,
                radius_radians,
                inclusive=False,
            )

            # calculate the areas per bin by the product
            # of pixel area by the number of pixels at each radial bin
            area[i] = hp.nside2pixarea(this_nside) * radial_bin_pixels.shape[0]

            # NOTE: select active pixels according to each radial bin
            bin_active_pixel_indexes = np.intersect1d(
                self._active_pixels[energy_id],
                radial_bin_pixels,
                return_indices=True,
            )[1]

            # obtain the excess, background, and expected excess at
            # each radial bin
            data = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()
            mdl = self._get_model_map(
                energy_id, n_point_sources, n_ext_sources
            ).as_partial()

            # select counts only from the pixels within specifid distance from
            # origin of radial profile
            bin_data = np.array([data[i] for i in bin_active_pixel_indexes])
            bin_bkg = np.array([bkg[i] for i in bin_active_pixel_indexes])
            bin_model = np.array([mdl[i] for i in bin_active_pixel_indexes])

            this_data_tot = np.sum(bin_data)
            this_bkg_tot = np.sum(bin_bkg)
            this_model_tot = np.sum(bin_model)

            background[i] = this_bkg_tot
            observation[i] = this_data_tot
            model[i] = this_model_tot
            signal[i] = this_data_tot - this_bkg_tot

        return area, signal, background, model

    def get_radial_profile(
        self,
        ra: float,
        dec: float,
        active_planes: list = None,
        max_radius: float = 3.0,
        n_radial_bins: int = 30,
        delta_step: float = 0.5,
        model_to_subtract: Optional[astromodels.Model | None] = None,
        subtract_model_from_model: bool = False,
    ):
        """Calculates radial profiles for a source in units of excess counts
           per steradian

        Args:
            ra (float): RA of origin of radial profile
            dec (float): Declincation of origin of radial profile
            active_planes (np.ndarray, optional): List of active planes over
            which to average. Defaults to None.
            max_radius (float, optional): Radius up to which evaluate the
            radial profile. Defaults to 3.0.
            n_radial_bins (int, optional): Number of radial bins to use for
            the profile. Defaults to 30.
            model_to_subtract (astromodels.model, optional): Another model to
            subtract from the data excess. Defaults to None.
            subtract_model_from_model (bool, optional): If True, and
            model_to_subtract is not None,
            subtract model from model too. Defaults to False.

        Returns:
            tuple(np.ndarray): returns list of radial distances, excess expected
            counts, excess counts, counts uncertainty, and list of sorted active_planes
        """
        # default is to use all active bins
        if active_planes is None:
            active_planes = self._active_planes

        # Make sure we use bins with data
        good_planes = [plane_id in active_planes for plane_id in self._active_planes]
        plane_ids = set(active_planes) & set(self._active_planes)

        # offset = 0.5
        offset = delta_step
        delta_r = 1.0 * max_radius / n_radial_bins
        radii = np.array([delta_r * (r + offset) for r in range(n_radial_bins)])

        # Get area of all pixels in a given circle
        # The area of each ring is then given by the difference between two
        # subsequent circe areas.
        area = np.array(
            [
                self.get_excess_background(ra, dec, r + offset * delta_r)[0]
                for r in radii
            ]
        )

        temp = area[1:] - area[:-1]
        area[1:] = temp

        # signals
        signal = np.array(
            [
                self.get_excess_background(ra, dec, r + offset * delta_r)[1]
                for r in radii
            ]
        )

        temp = signal[1:] - signal[:-1]
        signal[1:] = temp

        # backgrounds
        bkg = np.array(
            [
                self.get_excess_background(ra, dec, r + offset * delta_r)[2]
                for r in radii
            ]
        )

        temp = bkg[1:] - bkg[:-1]
        bkg[1:] = temp

        counts = signal + bkg

        # model
        # convert 'top hat' excess into 'ring' excesses.
        model = np.array(
            [
                self.get_excess_background(ra, dec, r + offset * delta_r)[3]
                for r in radii
            ]
        )

        temp = model[1:] - model[:-1]
        model[1:] = temp

        if model_to_subtract is not None:
            this_model = copy.deepcopy(self._likelihood_model)
            self.set_model(model_to_subtract)

            model_subtract = np.array(
                [
                    self.get_excess_background(ra, dec, r + offset * delta_r)[3]
                    for r in radii
                ]
            )

            temp = model_subtract[1:] - model_subtract[:-1]
            model_subtract[1:] = temp

            signal -= model_subtract

            if subtract_model_from_model:
                model -= model_subtract

            self.set_model(this_model)

        # NOTE: weights are calculated as expected number of gamma-rays/number
        # of background counts.here, use max_radius to evaluate the number of
        # gamma-rays/bkg counts. The weights do not depend on the radius,
        # but fill a matrix anyway so there's no confusion when multiplying
        # them to the data later. Weight is normalized (sum of weights over
        # the bins = 1).

        np.array(self.get_excess_background(ra, dec, max_radius)[1])[good_planes]

        total_bkg = np.array(self.get_excess_background(ra, dec, max_radius)[2])[
            good_planes
        ]

        total_model = np.array(self.get_excess_background(ra, dec, max_radius)[3])[
            good_planes
        ]

        w = np.divide(total_model, total_bkg)
        weight = np.array([w / np.sum(w) for _ in radii])

        # restrict profiles to the user-specified analysis bins
        area = area[:, good_planes]
        signal = signal[:, good_planes]
        model = model[:, good_planes]
        counts = counts[:, good_planes]
        bkg = bkg[:, good_planes]

        # average over the analysis bins
        excess_data = np.average(signal / area, weights=weight, axis=1)
        excess_error = np.sqrt(np.sum(counts * weight * weight / (area * area), axis=1))
        excess_model = np.average(model / area, weights=weight, axis=1)

        return radii, excess_model, excess_data, excess_error, sorted(plane_ids)

    def plot_radial_profile(
        self,
        ra: float,
        dec: float,
        active_planes: Optional[list[str] | None] = None,
        max_radius: float = 3.0,
        n_radial_bins: int = 30,
        delta_step: float = 0.8,
        model_to_subtract: Optional[astromodels.Model | None] = None,
        subtract_model_from_model: bool = False,
    ):
        """Plots radial profiles of data-background & model

        Args:
            ra (float): RA of origin of radial profile
            dec (float): Declination of origin of radial profile.
            active_planes (np.ndarray, optional): List of analysis bins over
            which to average.
            Defaults to None.
            max_radius (float, optional): Radius up to which the radial profile
            is evaluate; also used as the radius for the disk to calculate the
            gamma/hadron weights. Defaults to 3.0.
            n_radial_bins (int, optional): number of radial bins used for ring
            calculation. Defaults to 30.
            model_to_subtract (astromodels.model, optional): Another model that
            is to be subtracted from the data excess. Defaults to None.
            subtract_model_from_model (bool, optional): If True and
            model_to_subtract is not None, subtract from model too.
            Defaults to False.

        Returns:
            tuple(matplotlib.pyplot.Figure, pd.DataFrame): plot of data-background
            & model radial profile for source and a dataframe with all
            values for easy retrieval
        """

        (
            radii,
            excess_model,
            excess_data,
            excess_error,
            plane_ids,
        ) = self.get_radial_profile(
            ra,
            dec,
            active_planes,
            max_radius,
            n_radial_bins,
            delta_step,
            model_to_subtract,
            subtract_model_from_model,
        )

        # add a dataframe for easy retrieval for calculations of surface
        # brighntess, if necessary.
        df = pd.DataFrame(columns=["Excess", "Bkg", "Model"], index=radii)
        df.index.name = "Radii"
        df["Excess"] = excess_data
        df["Bkg"] = excess_error
        df["Model"] = excess_model

        fig, ax = plt.subplots(figsize=(10, 8))

        plt.errorbar(
            radii,
            excess_data,
            yerr=excess_error,
            capsize=0,
            color="black",
            label="Excess (data-bkg)",
            fmt=".",
        )

        plt.plot(radii, excess_model, color="red", label="Model")

        plt.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper right", numpoints=1, fontsize=16
        )
        plt.axhline(0, color="deepskyblue", linestyle="--")

        x_limits = [0, max_radius]
        plt.xlim(x_limits)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.ylabel(r"Apparent Radial Excess [sr$^{-1}$]", fontsize=18)
        plt.xlabel(
            f"Distance from source at ({ra:0.2f} $^{{\circ}}$, {dec:0.2f} $^{{\circ}}$)",
            fontsize=18,
        )

        if len(plane_ids) == 1:
            title = f"Radial Profile, bin {plane_ids[0]}"

        else:
            title = "Radial Profile"
            # tmptitle = f"Radial Profile, bins \n{plane_ids}"
            # width = 80
            # title = "\n".join(
            # tmptitle[i : i + width] for i in range(0, len(tmptitle), width)
            # )
            # title = tmptitle

        plt.title(title)

        ax.grid(True)

        with contextlib.suppress(Exception):
            plt.tight_layout()
        # try:
        #
        # plt.tight_layout()
        #
        # except Exception:
        #
        # pass

        return fig, df

    def display_spectrum(self):
        """
        Make a plot of the current spectrum and its residuals (integrated over space)

        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        total_counts = np.zeros(len(self._active_planes), dtype=float)
        total_model = np.zeros_like(total_counts)
        model_only = np.zeros_like(total_counts)
        net_counts = np.zeros_like(total_counts)
        yerr_low = np.zeros_like(total_counts)
        yerr_high = np.zeros_like(total_counts)

        for i, energy_id in enumerate(self._active_planes):
            data_analysis_bin: DataAnalysisBin = self._maptree[energy_id]

            this_model_map_hpx = self._get_expectation(
                data_analysis_bin, energy_id, n_point_sources, n_ext_sources
            )

            this_model_tot = np.sum(this_model_map_hpx)
            this_data_tot = np.sum(data_analysis_bin.observation_map.as_partial())
            this_bkg_tot = np.sum(data_analysis_bin.background_map.as_partial())

            total_counts[i] = this_data_tot
            net_counts[i] = this_data_tot - this_bkg_tot
            model_only[i] = this_model_tot

            this_wh_model = this_model_tot + this_bkg_tot
            total_model[i] = this_wh_model

            if this_data_tot >= 50.0:
                # Gaussian limit
                # Under the null hypothesis the data are distributed as a Gaussian with mu = model
                # and sigma = sqrt(model)
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                yerr_low[i] = np.sqrt(this_data_tot)
                yerr_high[i] = np.sqrt(this_data_tot)

            else:
                # Low-counts
                # Under the null hypothesis the data are distributed as a Poisson distribution with
                # mean = model, plot the 68% confidence interval (quantile=[0.16,1-0.16]).
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                quantile = 0.16
                mean = this_wh_model
                y_low = poisson.isf(1 - quantile, mu=mean)
                y_high = poisson.isf(quantile, mu=mean)
                yerr_low[i] = mean - y_low
                yerr_high[i] = y_high - mean

        residuals = old_div((total_counts - total_model), np.sqrt(total_model))
        residuals_err = [
            old_div(yerr_high, np.sqrt(total_model)),
            old_div(yerr_low, np.sqrt(total_model)),
        ]

        yerr = [yerr_high, yerr_low]

        return self._plot_spectrum(
            net_counts, yerr, model_only, residuals, residuals_err
        )

    def _plot_spectrum(self, net_counts, yerr, model_only, residuals, residuals_err):
        alice_style = copy.deepcopy(hep.style.ALICE)

        alice_style["xtick.top"] = False
        alice_style["ytick.right"] = False
        alice_style["xtick.minor.visible"] = False
        alice_style["ytick.minor.visible"] = True

        plt.style.use(alice_style)

        fig, subs = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, figsize=(14, 8)
        )
        planes = np.array(self._active_planes)
        subs[0].errorbar(
            planes,
            net_counts,
            yerr=yerr,
            capsize=0,
            color="black",
            label="Net counts",
            fmt=".",
        )

        subs[0].plot(planes, model_only, label="Convolved model")

        subs[0].legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", numpoints=1)

        # Residuals
        subs[1].axhline(0, linestyle="--")

        subs[1].errorbar(planes, residuals, yerr=residuals_err, capsize=0, fmt=".")

        y_limits = [min(net_counts[net_counts > 0]) / 2.0, max(net_counts) * 2.0]

        subs[0].set_yscale("log", nonpositive="clip")
        subs[0].set_ylabel("Counts per bin")
        subs[0].set_xticks([])

        subs[1].set_xlabel("Analysis bin")
        subs[1].set_ylabel(r"$\frac{{cts - mod - bkg}}{\sqrt{mod + bkg}}$")
        subs[1].set_xticks(planes)
        subs[1].set_xticklabels(self._active_planes, rotation=30)

        subs[0].set_ylim(y_limits)

        return fig

    def _process_bin(
        self,
        bin_id: str,
        this_data_analysis_bin: DataAnalysisBin,
        n_point_sources: int,
        n_ext_sources: int,
        bkg_renorm: float,
    ) -> tuple[str, float]:
        """
        Process a single bin, returning the expectation for the model.

        Parameters
        ----------
        bin_id: str
            The bin ID
        data_analysis_bin: DataAnalysisBin
            The data analysis bin
        n_point_sources: int
            The number of point sources
        n_ext_sources: int
            The number of extended sources
        bkg_renorm: float
            The background renormalization factor

        Returns
        -------
        tuple[str, float]
            Analysis bin id with its corresponding log-likelihood
        """

        this_model_map_hpx = self._get_expectation(
            this_data_analysis_bin,
            bin_id,
            n_point_sources,
            n_ext_sources,
        )

        obs = this_data_analysis_bin.observation_map.as_partial()
        bkg = this_data_analysis_bin.background_map.as_partial() * bkg_renorm

        this_pseudo_log_like = log_likelihood(obs, bkg, this_model_map_hpx)

        this_log_like = (
            this_pseudo_log_like
            - self._log_factorials[bin_id]
            - self._saturated_model_like_per_maptree[bin_id]
        )

        return bin_id, this_log_like

    def _multithreading_log_like(
        self,
        n_point_sources: int,
        n_ext_sources: int,
        bkg_renorm: float,
        n_jobs: int = 5,
    ) -> list[tuple[str, float]]:
        """Multithreading log-likelihood calculation for active bin in the analysis

        Parameters
        ----------
        n_point_sources : int
            Number of point sources present in model
        n_ext_sources : int
            Number of extended sources present in model
        bkg_renorm : float
            Background renormalization factor
        n_jobs : int, optional
            Number of threads to use, by default 5

        Returns
        -------
        list[tuple[str, float]]
            List of tuples with bin_id and log-likelihood
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from functools import partial

        # launches a thread pool with the number of threads specified by n_jobs
        with ThreadPoolExecutor(n_jobs) as pool:
            tasks = [
                pool.submit(
                    partial(
                        self._process_bin,
                        bin_id,
                        self._maptree[bin_id],
                        n_point_sources,
                        n_ext_sources,
                        bkg_renorm,
                    )
                )
                for bin_id in self._active_planes
            ]

        # processe the result of a thread as oon as it becomes available
        results = [future.result() for future in as_completed(tasks)]

        return results

    def get_log_like(
        self, individual_bins=False, return_null=False
    ) -> float | tuple[float, dict[str, float]]:
        """
        Return the value of the log-likelihood with the current values for the
        parameters.

        Parameters
        ----------
        individual_bins: bool
            If True, return the log-likelihood for each bin
        return_null: bool
            If True, return 0.0 instead of the log-likelihood
        """

        if return_null is True:
            n_point_sources: int = 0
            n_ext_sources: int = 0
        else:
            n_point_sources = self._likelihood_model.get_number_of_point_sources()  # type: ignore
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()  # type: ignore

            # Make sure that no source has been added since we filled the cache
            assert (
                n_point_sources == self._convolved_point_sources.n_sources_in_cache
                and n_ext_sources == self._convolved_ext_sources.n_sources_in_cache
            ), "The number of sources has changed. Please re-assign the model to the plugin."

        # This will hold the total log-likelihood
        total_log_like = 0
        log_like_per_bin: dict[str, float] = {}

        # Now compare with observation
        bkg_renorm = list(self._nuisance_parameters.values())[0].value
        multithread_log_like_results = self._multithreading_log_like(
            n_point_sources, n_ext_sources, bkg_renorm
        )
        total_log_like = sum(result[1] for result in multithread_log_like_results)
        if individual_bins is True:
            log_like_per_bin = dict(multithread_log_like_results)

        # for bin_id in self._active_planes:
        #     this_data_analysis_bin: DataAnalysisBin = self._maptree[bin_id]

        #     this_model_map_hpx = self._get_expectation(
        #         this_data_analysis_bin,
        #         bin_id,
        #         n_point_sources,
        #         n_ext_sources,
        #     )

        #     obs = this_data_analysis_bin.observation_map.as_partial()
        #     bkg = this_data_analysis_bin.background_map.as_partial() * bkg_renorm

        #     this_pseudo_log_like = log_likelihood(obs, bkg, this_model_map_hpx)

        #     this_log_like = (
        #         this_pseudo_log_like
        #         - self._log_factorials[bin_id]
        #         - self._saturated_model_like_per_maptree[bin_id]
        #     )

        #     total_log_like += this_log_like

        #     if individual_bins is True:
        #         log_like_per_bin[bin_id] = this_log_like

        if individual_bins is True:
            for k in log_like_per_bin:
                log_like_per_bin[k] /= total_log_like
            return total_log_like, log_like_per_bin

        return total_log_like

    def write(self, response_file_name, map_tree_file_name):
        """
        Write this dataset to disk in HDF format.

        :param response_file_name: filename for the response
        :param map_tree_file_name: filename for the map tree
        :return: None
        """

        self._maptree.write(map_tree_file_name)
        self._response.write(response_file_name)

    def get_simulated_dataset(self, name):
        """
        Return a simulation of this dataset using the current model with current parameters.

        :param name: new name for the new plugin instance
        :return: a HAL instance
        """

        # First get expectation under the current model and store them, if we didn't do it yet

        if self._clone is None:
            n_point_sources = self._likelihood_model.get_number_of_point_sources()
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

            expectations = collections.OrderedDict()

            for bin_id in self._maptree:
                data_analysis_bin = self._maptree[bin_id]
                if bin_id not in self._active_planes:
                    expectations[bin_id] = None

                else:
                    expectations[bin_id] = (
                        self._get_expectation(
                            data_analysis_bin, bin_id, n_point_sources, n_ext_sources
                        )
                        + data_analysis_bin.background_map.as_partial()
                    )

            if parallel_client.is_parallel_computation_active():
                # Do not clone, as the parallel environment already makes clones

                clone = self

            else:
                clone = copy.deepcopy(self)

            self._clone = (clone, expectations)

        # Substitute the observation and background for each data analysis bin
        for bin_id in self._clone[0]._maptree:
            data_analysis_bin = self._clone[0]._maptree[bin_id]

            if bin_id not in self._active_planes:
                continue

            else:
                # Active plane. Generate new data
                expectation = self._clone[1][bin_id]
                new_data = np.random.poisson(
                    expectation, size=(1, expectation.shape[0])
                ).flatten()

                # Substitute data
                data_analysis_bin.observation_map.set_new_values(new_data)

        # Now change name and return
        self._clone[0]._name = name
        # Adjust the name of the nuisance parameter
        old_name = list(self._clone[0]._nuisance_parameters.keys())[0]
        new_name = old_name.replace(self.name, name)
        self._clone[0]._nuisance_parameters[new_name] = self._clone[
            0
        ]._nuisance_parameters.pop(old_name)

        # Recompute biases
        self._clone[0]._compute_likelihood_biases()

        return self._clone[0]

    @staticmethod
    def _calculate_point_source_expectation(
        pts_id: int,
        energy_bin_id: str,
        data_analysis_bin: DataAnalysisBin,
        convolved_source_container: ConvolvedSourcesContainer,
        lock: RLock,
        psf_integration_method: str = "exact",
    ) -> ndarray:
        """
        Compute the expected counts for a point source

        Parameters
        ----------
        pts_id : int
            Assigned id of the point source
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        data_analysis_bin : DataAnalysisBin
            Data analysis bin with observed counts and background
        convolved_source_container : ConvolvedSourcesContainer
            Container with convolved point sources
        lock : RLock
            Lock to use for thread safety
        psf_integration_method : str
            Method to use for PSF integration. Options are 'exact' and 'fast'

        Returns
        -------
        NDArray[np.float64]
            Expected counts for a point source
        """
        this_conv_src: ConvolvedPointSource = convolved_source_container[pts_id]

        # ? multithreading for point sources causes racing conditions,
        # ? not sure what is going on and this requires more of an in-depth look
        # ? For now, a lock is put in place to prevent such conditions
        with lock:
            expectation_per_transit = this_conv_src.get_source_map(
                energy_bin_id,
                tag=None,
                psf_integration_method=psf_integration_method,
            )

        expectation_from_this_source = (
            expectation_per_transit * data_analysis_bin.n_transits
        )

        return expectation_from_this_source

    @staticmethod
    def _worker_func(
        energy_bin_id: str,
        convolved_source: ConvolvedExtendedSource2D | ConvolvedExtendedSource3D,
    ) -> ndarray:
        """Utility function to evaluate the expected counts from an extended source

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        convolved_source : ConvolvedExtendedSource2D | ConvolvedExtendedSource3D
            Convolved extended source

        Returns
        -------
        ndarray
            Convolved extended source image
        """
        return convolved_source.get_source_map(energy_bin_id)

    @staticmethod
    def _extended_source_expectation(
        energy_bin_id: str,
        n_ext_sources: int,
        worker_func: Callable[
            [str, ConvolvedExtendedSource2D | ConvolvedExtendedSource3D], ndarray
        ],
        convolved_source_container: ConvolvedSourcesContainer,
        n_jobs: int = 5,
    ) -> ndarray:
        """Calculate the expected counts from extended sources

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        n_ext_sources : int
            Number of extended sources present in the model
        worker_func : Callable[[str, ConvolvedExtendedSource2D | ConvolvedExtendedSource3D], ndarray]
            Function to calculate the expected counts from an extended source
        convolved_source_container : ConvolvedSourcesContainer
            Container with convolved extended sources
        n_jobs : int, optional
            Number of threads to use, by default 5

        Returns
        -------
        ndarray
            Expected counts from extended sources for convolution with the PSF
        """
        extended_source_map = None

        # ? Can this for loop be parellelized?
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from functools import partial

        with ThreadPoolExecutor(n_jobs) as executor:
            tasks = [
                executor.submit(
                    partial(
                        worker_func,
                        energy_bin_id,
                        convolved_source_container[ext_id],
                    )
                )
                for ext_id in range(n_ext_sources)
            ]
        extended_source_map = np.sum(
            [future.result() for future in as_completed(tasks)], axis=0
        )
        # ? Leaving this code here for more thorough testing
        # for ext_id in range(n_ext_sources):
        #     this_cnv_source: ConvolvedExtendedSource2D | ConvolvedExtendedSource3D = (
        #         convolved_source_container[ext_id]
        #     )

        #     this_ext_source = this_cnv_source.get_source_map(energy_bin_id)

        #     if extended_source_map is None:
        #         extended_source_map = this_ext_source
        #     else:
        #         extended_source_map += this_ext_source

        return extended_source_map

    @staticmethod
    def _get_model_map_from_ext_source(
        energy_bin_id: str,
        psf_convolutors: dict[str, PSFConvolutor],
        data_analysis_bin: DataAnalysisBin,
        extended_src_model_map: NDArray[np.float64],
    ) -> ndarray:
        """Calculate the model map for an extended source by convolving with the PSF

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        psf_convolutors : dict[str, PSFConvolutor]
            Dictionary with the PSF convolutors for each analysis bin
        data_analysis_bin : DataAnalysisBin
            Observation analysis bin with observed counts and background
        extended_src_model_map : NDArray[np.float64]
            Map with expected counts per transit for extended sources within model

        Returns
        -------
        NDArray[np.float64]
            Returns the expected counts using the current state of the likelihood
        """
        return (
            psf_convolutors[energy_bin_id].extended_source_image(extended_src_model_map)
            * data_analysis_bin.n_transits
        )

    def _get_expectation(
        self,
        data_analysis_bin: DataAnalysisBin,
        energy_bin_id: str,
        n_point_sources: int,
        n_ext_sources: int,
    ) -> ndarray | float:
        """Calculate the model map for a given analysis bin with the number of
        sources present in the model

        Parameters
        ----------
        data_analysis_bin : DataAnalysisBin
            Observation analysis bin with observed counts and background
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        n_point_sources : int
            Number of point sources present within the likelihood model instance
        n_ext_sources : int
            Number of extended sources present within the likelihood model instance

        Returns
        -------
        NDArray[np.float64]
            Returns the expected counts using the current state of the likelihood
            model instance.
        """

        # Compute the expectation from the model
        this_model_map = None

        # first process the point sources
        if n_point_sources > 0:
            # ? Can this for loop be parellelized?
            for pts_id in range(n_point_sources):
                expectation_from_this_source = self._calculate_point_source_expectation(
                    pts_id,
                    energy_bin_id,
                    data_analysis_bin,
                    self._convolved_point_sources,
                    self.lock,
                    self._psf_integration_method,
                )

                if this_model_map is None:
                    this_model_map = expectation_from_this_source
                else:
                    this_model_map += expectation_from_this_source

        # Now process extended sources
        if n_ext_sources > 0:
            # this_ext_model_map = None

            tot_ext_sources_expectation_per_transit = self._extended_source_expectation(
                energy_bin_id,
                n_ext_sources,
                self._worker_func,
                self._convolved_ext_sources,
            )

            # Now convolve with the PSF
            extended_source_expectation = self._get_model_map_from_ext_source(
                energy_bin_id,
                self._psf_convolutors,
                data_analysis_bin,
                tot_ext_sources_expectation_per_transit,
            )

            if this_model_map is None:
                this_model_map = extended_source_expectation
            else:
                this_model_map += extended_source_expectation

        if this_model_map is not None:
            # Now transform from the flat sky projection to HEALPiX

            this_model_map_hpx = self._convert_from_flat_to_hpx(
                energy_bin_id,
                this_model_map,
                self._flat_sky_projection,
                data_analysis_bin,
                self._flat_sky_to_healpix_transform,
            )

        else:
            # No sources

            this_model_map_hpx = 0.0

        return this_model_map_hpx

    @staticmethod
    def _convert_from_flat_to_hpx(
        energy_bin_id: str,
        model_map: NDArray[np.float64],
        flat_sky_projection: FlatSkyProjection,
        data_analysis_bin: DataAnalysisBin,
        flat_sky_to_healpix_transform: dict[str, FlatSkyToHealpixTransform],
    ) -> ndarray:
        """Convert a flat sky projection to a HEALPiX projection

        Parameters
        ----------
        energy_bin_id : str
            Analysis bin defined from maptree and response function
        model_map : NDArray[np.float64]
            Map with expected counts per transit for sources within model
        flat_sky_projection : FlatSkyProjection
            Flat sky projection of the data analysis bin
        data_analysis_bin : DataAnalysisBin
            Observation analysis bin with observed counts and background
        flat_sky_to_healpix_transform : dict[str, FlatSkyToHealpixTransform]
            Dictionary with the flat sky to HEALPiX transform for each analysis bin

        Returns
        -------
        NDArray[np.float64]
            Returns the expected counts model map in HEALPiX projection
        """

        # First divide for the pixel area because we need to interpolate brightness

        model_map = np.divide(model_map, flat_sky_projection.project_plane_pixel_area)

        model_map_hpx = flat_sky_to_healpix_transform[energy_bin_id](
            model_map, fill_value=0.0
        )

        # Now multiply by the pixel area of the new map to go back to flux
        model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)

        return model_map_hpx

    @staticmethod
    def _represent_healpix_map(
        fig, hpx_map, longitude, latitude, xsize, resolution, smoothing_kernel_sigma
    ):
        proj = get_gnomonic_projection(
            fig, hpx_map, rot=(longitude, latitude, 0.0), xsize=xsize, reso=resolution
        )

        if smoothing_kernel_sigma is not None:
            # Get the sigma in pixels
            sigma = old_div(smoothing_kernel_sigma * 60, resolution)

            proj = convolve(
                list(proj),
                Gaussian2DKernel(sigma),
                nan_treatment="fill",
                preserve_nan=True,
            )

        return proj

    def display_fit(self, smoothing_kernel_sigma=0.1, display_colorbar=False):
        """
        Make a figure containing 4 maps for each active analysis bins with respectively model, data,
        background and residuals. The model, data and residual maps are smoothed, the background
        map is not.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel, for all but
        background maps
        :param display_colorbar: whether or not to display the colorbar in the residuals
        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # This is the resolution (i.e., the size of one pixel) of the image
        resolution = 3.0  # arcmin

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        n_active_planes = len(self._active_planes)
        n_columns = 4

        fig, subs = plt.subplots(
            n_active_planes,
            n_columns,
            figsize=(2.7 * n_columns, n_active_planes * 2),
            squeeze=False,
        )

        prog_bar = tqdm(total=len(self._active_planes), desc="Smoothing planes")

        images = ["None"] * n_columns

        for i, plane_id in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[plane_id]

            # Get the center of the projection for this plane
            this_ra, this_dec = self._roi.ra_dec_center

            # Make a full healpix map for a second
            whole_map = self._get_model_map(
                plane_id, n_point_sources, n_ext_sources
            ).as_dense()

            # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
            longitude = ra_to_longitude(this_ra)

            # Declination is already between -90 and 90
            latitude = this_dec

            # Background and excess maps
            bkg_subtracted, _, background_map = self._get_excess(
                data_analysis_bin, all_maps=True
            )

            # Make all the projections: model, excess, background, residuals
            proj_model = self._represent_healpix_map(
                fig,
                whole_map,
                longitude,
                latitude,
                xsize,
                resolution,
                smoothing_kernel_sigma,
            )
            # Here we removed the background otherwise nothing is visible
            # Get background (which is in a way "part of the model" since the uncertainties are neglected)
            proj_data = self._represent_healpix_map(
                fig,
                bkg_subtracted,
                longitude,
                latitude,
                xsize,
                resolution,
                smoothing_kernel_sigma,
            )
            # No smoothing for this one (because a goal is to check it is smooth).
            proj_bkg = self._represent_healpix_map(
                fig, background_map, longitude, latitude, xsize, resolution, None
            )
            proj_residuals = proj_data - proj_model

            # Common color scale range for model and excess maps
            vmin = min(np.nanmin(proj_model), np.nanmin(proj_data))
            vmax = max(np.nanmax(proj_model), np.nanmax(proj_data))

            # FIXME: Change the following to f-strings
            # Plot model
            images[0] = subs[i][0].imshow(
                proj_model, origin="lower", vmin=vmin, vmax=vmax
            )
            subs[i][0].set_title("model, bin {}".format(data_analysis_bin.name))

            # Plot data map
            images[1] = subs[i][1].imshow(
                proj_data, origin="lower", vmin=vmin, vmax=vmax
            )
            subs[i][1].set_title("excess, bin {}".format(data_analysis_bin.name))

            # Plot background map.
            images[2] = subs[i][2].imshow(proj_bkg, origin="lower")
            subs[i][2].set_title("background, bin {}".format(data_analysis_bin.name))

            # Now residuals
            images[3] = subs[i][3].imshow(proj_residuals, origin="lower")
            subs[i][3].set_title("residuals, bin {}".format(data_analysis_bin.name))

            # Remove numbers from axis
            for j in range(n_columns):
                subs[i][j].axis("off")

            if display_colorbar:
                for j, image in enumerate(images):
                    plt.colorbar(image, ax=subs[i][j])

            prog_bar.update(1)

        # fig.set_tight_layout(True)
        fig.set_layout_engine("tight")

        return fig

    def _get_optimal_xsize(self, resolution):
        return 2.2 * self._roi.data_radius.to("deg").value / (resolution / 60.0)

    def display_stacked_image(self, smoothing_kernel_sigma=0.5):
        """
        Display a map with all active analysis bins stacked together.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel to apply
        :return: a matplotlib.Figure instance
        """

        # This is the resolution (i.e., the size of one pixel) of the image in arcmin
        resolution = 3.0

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        active_planes_bins = [self._maptree[x] for x in self._active_planes]

        # Get the center of the projection for this plane
        this_ra, this_dec = self._roi.ra_dec_center

        # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
        longitude = ra_to_longitude(this_ra)

        # Declination is already between -90 and 90
        latitude = this_dec

        total = None

        for i, data_analysis_bin in enumerate(active_planes_bins):
            # Plot data
            background_map = data_analysis_bin.background_map.as_dense()
            this_data = data_analysis_bin.observation_map.as_dense() - background_map
            idx = np.isnan(this_data)
            # this_data[idx] = hp.UNSEEN

            if i == 0:
                total = this_data

            else:
                # Sum only when there is no UNSEEN, so that the UNSEEN pixels will stay UNSEEN
                total[~idx] += this_data[~idx]

        delta_coord = (self._roi.data_radius.to("deg").value * 2.0) / 15.0

        fig, sub = plt.subplots(1, 1)

        proj = self._represent_healpix_map(
            fig, total, longitude, latitude, xsize, resolution, smoothing_kernel_sigma
        )

        cax = sub.imshow(proj, origin="lower")
        fig.colorbar(cax)
        sub.axis("off")

        hp.graticule(delta_coord, delta_coord)

        return fig

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()

    def get_number_of_data_points(self):
        """
        Return the number of active bins across all active analysis bins

        :return: number of active bins
        """

        n_points = 0

        for bin_id in self._maptree:
            n_points += self._maptree[bin_id].observation_map.as_partial().shape[0]

        return n_points

    def _get_model_map(self, plane_id, n_pt_src, n_ext_src):
        """
        This function returns a model map for a particular bin
        """

        if plane_id not in self._active_planes:
            raise ValueError(f"{plane_id} not a plane in the current model")

        model_map = SparseHealpix(
            self._get_expectation(
                self._maptree[plane_id], plane_id, n_pt_src, n_ext_src
            ),
            self._active_pixels[plane_id],
            self._maptree[plane_id].observation_map.nside,
        )

        return model_map

    def _get_excess(self, data_analysis_bin, all_maps=True):
        """
        This function returns the excess counts for a particular bin
        if all_maps=True, also returns the data and background maps
        """
        data_map = data_analysis_bin.observation_map.as_dense()
        bkg_map = data_analysis_bin.background_map.as_dense()
        excess = data_map - bkg_map

        if all_maps:
            return excess, data_map, bkg_map
        return excess

    def _write_a_map(self, file_name, which, fluctuate=False, return_map=False):
        """
        This writes either a model map or a residual map, depending on which one is preferred
        """
        which = which.lower()
        assert which in ["model", "residual"]

        n_pt = self._likelihood_model.get_number_of_point_sources()
        n_ext = self._likelihood_model.get_number_of_extended_sources()

        map_analysis_bins = collections.OrderedDict()

        if fluctuate:
            poisson_set = self.get_simulated_dataset("model map")

        # TODO: parallelize this for loop
        for plane_id in self._active_planes:
            data_analysis_bin = self._maptree[plane_id]

            bkg = data_analysis_bin.background_map
            obs = data_analysis_bin.observation_map

            if fluctuate:
                model_excess = (
                    poisson_set._maptree[plane_id].observation_map
                    - poisson_set._maptree[plane_id].background_map
                )
            else:
                model_excess = self._get_model_map(plane_id, n_pt, n_ext)

            if which == "residual":
                bkg += model_excess

            if which == "model":
                obs = model_excess + bkg

            this_bin = DataAnalysisBin(
                plane_id,
                observation_hpx_map=obs,
                background_hpx_map=bkg,
                active_pixels_ids=self._active_pixels[plane_id],
                n_transits=data_analysis_bin.n_transits,
                scheme="RING",
            )

            map_analysis_bins[plane_id] = this_bin

        # save the file
        new_map_tree = MapTree(map_analysis_bins, self._roi)
        new_map_tree.write(file_name)

        if return_map:
            return new_map_tree

    def write_model_map(
        self, file_name, poisson_fluctuate=False, test_return_map=False
    ):
        """
        This function writes the model map to a file.
        The interface is based off of HAWCLike for consistency
        """
        if test_return_map:
            log.warning(
                "test_return_map=True should only be used for testing purposes!"
            )
        return self._write_a_map(file_name, "model", poisson_fluctuate, test_return_map)

    def write_residual_map(self, file_name, test_return_map=False):
        """
        This function writes the residual map to a file.
        The interface is based off of HAWCLike for consistency
        """
        if test_return_map:
            log.warning(
                "test_return_map=True should only be used for testing purposes!"
            )
        return self._write_a_map(file_name, "residual", False, test_return_map)
