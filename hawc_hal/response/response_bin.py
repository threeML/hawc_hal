""" Generate ResponseBin for HAWC Likelihood plugin"""
from dataclasses import dataclass, field

import boost_histogram as bh
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Self

from ..psf_fast import InvalidPSF, InvalidPSFError, PSFWrapper

# from typing import Self # available on python 3.11+
# NOTE: definition of a few constants to be used thorought the module
LOG_BASE: int = 10
ndarray = NDArray[np.float64]
dataframe = pd.DataFrame


@dataclass
class EnergyBin:
    """
    Constructor arguments:
    :param bin_name: Name of the analysis bin
    :param signal_events: Simulated signal events
    :param bkg_events: Simulated background events
    :param log_log_params: Best-fit parameters obtained from response file
    :param log_log_shape: Spectral shape
    """

    bin_name: str
    signal_events: bh.Histogram
    bkg_events: bh.Histogram
    log_log_params: np.ndarray
    log_log_shape: str

    # Variables to be initialized with the _get_edges method
    _lower_edges: ndarray = field(init=False)
    _centers: ndarray = field(init=False)
    _upper_edges: ndarray = field(init=False)

    def _log_log_spectrum(self, log_energy: float, log_log_shape: str) -> float:
        """Evaluate the differential flux from log10(simulated energy) values

        :param log_energy: simulated energy in log10 scale units (TeV)
        :type log_energy: float
        :param log_log_shape: best-fit paraemters obtained from response file
        :type log_log_shape: str
        :raises ValueError: There is an unknown spectral shape provided
        :return: Differential flux in units (TeV^-1 cm^-2 s^-1) in log10 scale
        """
        parameters = self.log_log_params

        if log_log_shape == "SimplePowerLaw":
            return np.log10(parameters[0]) - parameters[1] * log_energy

        if log_log_shape == "CutOffPowerLaw":
            return (
                np.log10(parameters[0])
                - parameters[1] * log_energy
                - np.log10(np.exp(1.0))
                * np.power(10.0, log_energy - np.log10(parameters[2]))
            )

        raise ValueError("Unknown spectral shape.")

    def _get_edges(self) -> None:
        """Read the lower, center and upper edges of the energy bins"""
        self._lower_edges = np.array(self.signal_events.axes.edges[0][:-1])
        self._centers = np.array(self.signal_events.axes.centers[0])
        self._upper_edges = np.array(self.signal_events.axes.edges[0][1:])

    @property
    def get_differential_fluxes(self) -> np.ndarray:
        """Calculate the differential fluxes with log energy values"""
        self._get_edges()
        differential_fluxes = np.array(
            [
                self._log_log_spectrum(log_energy, self.log_log_shape)
                for log_energy in self._centers
            ]
        )

        return LOG_BASE**differential_fluxes

    @property
    def get_energy_bin_low(self) -> np.ndarray:
        """Energy bins lower edge"""
        return LOG_BASE**self._lower_edges

    @property
    def get_energy_bin_upper(self) -> np.ndarray:
        """Energy bins upper edge"""
        return LOG_BASE**self._upper_edges

    @property
    def get_energy_bin_centers(self) -> np.ndarray:
        """Energy bins center edge"""
        return LOG_BASE**self._centers

    @property
    def get_signal_events(self) -> np.ndarray:
        """Retrieve simulated signal events"""
        return self.signal_events.values()

    @property
    def get_bkg_events(self) -> np.ndarray:
        """Retrieve simulated background events"""
        return self.bkg_events.values()

    @property
    def get_bin_name(self) -> str:
        """Retrieve name of analysis name"""
        return self.bin_name


class ResponseBin:
    """
    Stores detector response for one declination band and one analysis
    bin (called "name" or "analysis_bin_id" below).
    """

    def __init__(
        self,
        name,
        min_dec,
        max_dec,
        dec_center,
        sim_n_sig_events,
        sim_n_bg_events,
        sim_energy_bin_low,
        sim_energy_bin_centers,
        sim_energy_bin_hi,
        sim_differential_photon_fluxes,
        sim_signal_events_per_bin,
        psf,
    ):
        self._name = name
        self._min_dec = min_dec
        self._max_dec = max_dec
        self._dec_center = dec_center
        self._sim_n_sig_events = sim_n_sig_events
        self._sim_n_bg_events = sim_n_bg_events
        self._sim_energy_bin_low = sim_energy_bin_low
        self._sim_energy_bin_centers = sim_energy_bin_centers
        self._sim_energy_bin_hi = sim_energy_bin_hi
        self._sim_differential_photon_fluxes = sim_differential_photon_fluxes
        self._sim_signal_events_per_bin = sim_signal_events_per_bin
        self._psf = psf  # type: PSFWrapper

    # ? In the middle of refactoring some of the code, so this might or might
    # ? not be useful
    # @staticmethod
    # def _get_en_th1d(
    #     open_ttree: uproot.ReadOnlyDirectory,
    #     dec_id: int,
    #     analysis_bin_id: str,
    #     prefix: str,
    # ):
    #     en_sig_label = f"En{prefix}_dec{dec_id}_nh{analysis_bin_id}"

    #     hist_prefix = f"dec_{dec_id:02d}/nh_{analysis_bin_id}/{en_sig_label}"
    #     if open_ttree.get(hist_prefix, None) is None:
    #         hist_prefix = f"dec_{dec_id:02d}/nh_0{analysis_bin_id}/{en_sig_label}"

    #     this_en_th1d = open_ttree[hist_prefix].to_hist()

    #     if not this_en_th1d:
    #         raise IOError(f"Could not find TH1D named {en_sig_label}.")

    #     return this_en_th1d

    @classmethod
    def from_ttree(
        cls,
        analysis_bin_id: str,
        energy_hist: bh.Histogram,
        energy_hist_bkg: bh.Histogram,
        psf_fit_params: ndarray,
        log_log_params: ndarray,
        log_log_shape: str,
        min_dec: ndarray,
        dec_center: ndarray,
        max_dec: ndarray,
    ) -> Self:
        """Read in the response file with ROOT format and organize all the necessary
        information

        :param analysis_bin_id: Active analysis bin name defined in the response file
        :param energy_hist: Energy histogram from response file
        :param energy_hist_bkg: Background energy histogram from response file
        :param psf_fit_params: Best-fit PSF parameters for the active analysis_bin
        :param log_log_params: Best-fit params of the LogLogSpectrum TF1
        :param log_log_shape: Name of the spectral shape used to fit the PSF TF1
        :param min_dec: Lower declination edges
        :param dec_center: Declination bin centers
        :param max_dec: Upper declination edges
        :return: ResponseBin object with all the necessary information for a given analysis_bin
        :rtype: ResponseBin
        """

        # NOTE: Load all the information to the EnergyBin class for each bin_name
        # and for the complex processing of the differential flux,
        # energy lower_edges, centers, upper_edges
        energy_bin = EnergyBin(
            analysis_bin_id, energy_hist, energy_hist_bkg, log_log_params, log_log_shape
        )

        # Now let's see what has been simulated, i.e., the differential flux
        # at the center of each bin_name of the en_sig histogram
        sim_differential_photon_fluxes = energy_bin.get_differential_fluxes
        sim_energy_bin_low = energy_bin.get_energy_bin_low
        sim_energy_bin_centers = energy_bin.get_energy_bin_centers
        sim_energy_bin_high = energy_bin.get_energy_bin_upper

        sim_signal_events_per_bin = energy_bin.get_signal_events

        # Now read the various TF1(s) for PSF, signal and background
        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_sig_events = energy_bin.get_signal_events.sum()
        sim_n_bg_events = energy_bin.get_bkg_events.sum()

        # NOTE: uproot doesn't have the ability to read and evaluate TF1
        # but we pass the psf_params to the PSFWrapper class which can
        # evaluate the PSF function for us
        psf_fun = PSFWrapper.psf_eval(psf_fit_params)

        return cls(
            analysis_bin_id,
            min_dec,
            max_dec,
            dec_center,
            sim_n_sig_events,
            sim_n_bg_events,
            sim_energy_bin_low,
            sim_energy_bin_centers,
            sim_energy_bin_high,
            sim_differential_photon_fluxes,
            sim_signal_events_per_bin,
            psf_fun,
        )

    def to_pandas(self) -> tuple[dataframe, dict[str, ndarray], dataframe]:
        """Organizes information a response ROOT file into a dataframe for later storage
        in a file with HDF5 format

        :return: Tuple that contains essential information from the response file:
        simulated declination bins, signal events and simulated background events,
        and the expected counts from the PSF
        """

        # In the metadata let's save all single values (floats)
        meta: dict[str, ndarray] = {
            "min_dec": self._min_dec,
            "max_dec": self._max_dec,
            "declination_center": self._dec_center,
            "n_sim_signal_events": self._sim_n_sig_events,
            "n_sim_bkg_events": self._sim_n_bg_events,
        }

        # Now make a dataframe containing the elements of the simulation
        items = (
            ("sim_energy_bin_low", pd.Series(self.sim_energy_bin_low)),
            ("sim_energy_bin_centers", pd.Series(self.sim_energy_bin_centers)),
            ("sim_energy_bin_hi", pd.Series(self.sim_energy_bin_hi)),
            (
                "sim_differential_photon_fluxes",
                pd.Series(self.sim_differential_photon_fluxes),
            ),
            ("sim_signal_events_per_bin", pd.Series(self.sim_signal_events_per_bin)),
        )

        df = pd.DataFrame.from_dict(dict(items))

        return df, meta, self.psf.to_pandas()

    def combine_with_weights(
        self, other_response_bin: Self, dec_center: float, w1: ndarray, w2: ndarray
    ):
        """
        Produce another response bin which is the weighted sum of this one and
        the other one passed.

        :param other_response_bin: another response bin for which to evaluate
        the weighted sum
        :param w1: weights for current instance of response bin
        :param w2: weights for other instance of response bin
        :return: New instance of response bin with combined weights with combined
        weights
        """

        assert np.isclose(w1 + w2, 1.0), "Weights are not properly normalized"

        new_name = "interpolated_%s" % self._name

        # Use np.nan as declination boundaries to indicate that this is actually interpolated
        min_dec, max_dec = np.nan, np.nan

        n_sim_signal_events = (
            w1 * self._sim_n_sig_events + w2 * other_response_bin._sim_n_sig_events
        )
        n_sim_bkg_events = (
            w1 * self._sim_n_bg_events + w2 * other_response_bin._sim_n_bg_events
        )

        # We assume that the bin centers are the same
        assert np.allclose(
            self._sim_energy_bin_centers, other_response_bin._sim_energy_bin_centers
        )

        sim_differential_photon_fluxes = (
            w1 * self._sim_differential_photon_fluxes
            + w2 * other_response_bin._sim_differential_photon_fluxes
        )

        sim_signal_events_per_bin = (
            w1 * self._sim_signal_events_per_bin
            + w2 * other_response_bin._sim_signal_events_per_bin
        )

        # Now interpolate the psf, if none is invalid
        try:
            new_psf = self._psf.combine_with_other_psf(other_response_bin._psf, w1, w2)
        except InvalidPSFError:
            new_psf = InvalidPSF()

        new_response_bin = ResponseBin(
            new_name,
            min_dec,
            max_dec,
            dec_center,
            n_sim_signal_events,
            n_sim_bkg_events,
            self._sim_energy_bin_low,
            self._sim_energy_bin_centers,
            self._sim_energy_bin_hi,
            sim_differential_photon_fluxes,
            sim_signal_events_per_bin,
            new_psf,
        )

        return new_response_bin  # type: ResponseBin

    @property
    def name(self):
        return self._name

    @property
    def declination_boundaries(self):
        return (self._min_dec, self._max_dec)

    @property
    def declination_center(self):
        return self._dec_center

    @property
    def psf(self):
        return self._psf

    @property
    def n_sim_signal_events(self):
        return self._sim_n_sig_events

    @property
    def n_sim_bkg_events(self):
        return self._sim_n_bg_events

    @property
    def sim_energy_bin_low(self):
        return self._sim_energy_bin_low

    @property
    def sim_energy_bin_centers(self):
        return self._sim_energy_bin_centers

    @property
    def sim_energy_bin_hi(self):
        return self._sim_energy_bin_hi

    @property
    def sim_differential_photon_fluxes(self):
        return self._sim_differential_photon_fluxes

    @property
    def sim_signal_events_per_bin(self):
        return self._sim_signal_events_per_bin
