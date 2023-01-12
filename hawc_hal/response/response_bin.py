from builtins import range
from builtins import object
import uproot
import numpy as np
import pandas as pd

from ..psf_fast import PSFWrapper, InvalidPSF, InvalidPSFError


class ResponseBin(object):
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

    @staticmethod
    def _get_en_th1d(
        open_ttree: uproot.ReadOnlyDirectory,
        dec_id: int,
        analysis_bin_id: str,
        prefix: str,
    ):

        en_sig_label = f"En{prefix}_dec{dec_id}_nh{analysis_bin_id}"

        try:
            hist_prefix = f"dec_{dec_id:02d}/nh_{analysis_bin_id}/{en_sig_label}"
            this_en_th1d = open_ttree[hist_prefix].to_hist()

        except uproot.KeyInFileError:

            hist_prefix = f"dec_{dec_id:02d}/nh_0{analysis_bin_id}/{en_sig_label}"
            this_en_th1d = open_ttree[hist_prefix].to_hist()

        if not this_en_th1d:

            raise IOError(f"Could not find TH1D named {en_sig_label}.")

        return this_en_th1d

    @classmethod
    def from_ttree(
        cls,
        root_file: uproot.ReadOnlyDirectory,
        dec_id: int,
        analysis_bin_id: str,
        log_log_params: np.ndarray,
        log_log_shape: str,
        min_dec: np.ndarray,
        dec_center: np.ndarray,
        max_dec: np.ndarray,
    ):
        """
        Obtain the information from Response ROOT file

        Args:
            root_file (object): ROOT object reading information with uproot functionality
            dec_id (int): declination band id
            analysis_bin_id (str): data analysis name
            log_log_params (np.ndarray): params from LogLogSpectrum TF1
            min_dec (np.ndarray): numpy array with lower declination bin edges
            dec_center (np.ndarray): numpy array with declination center values
            max_dec (np.ndarray): numpy array with upper declination bin edges
        """

        def log_log_spectrum(log_energy: float, parameters: np.ndarray):
            """Evaluate the differential flux from log10(simulate energy) values

            Args:
                log_energy (float): simulated energy in log10 scale
                parameters (np.ndarray): best-fit parameters obtained from response file

            Returns:
                float: returns differential flux in units (TeV^-1 cm^-2 s^-1) in log10 scale
            """

            if log_log_shape == "SimplePowerLaw":

                return np.log10(parameters[0]) - parameters[1] * log_energy

            if log_log_shape == "CutOffPowerLaw":

                return (
                    np.log10(parameters[0])
                    - parameters[1] * log_energy
                    - np.log10(np.exp(1.0)) * np.power(10.0, log_energy - np.log10(parameters[2]))
                )

            else:
                raise ValueError("Unknown spectral shape.")

        this_en_sig_th1d = cls._get_en_th1d(root_file, dec_id, analysis_bin_id, prefix="Sig")

        this_en_bg_th1d = cls._get_en_th1d(root_file, dec_id, analysis_bin_id, prefix="Bg")

        total_bins = this_en_sig_th1d.shape[0]
        sim_energy_bin_low = np.zeros(total_bins)
        sim_energy_bin_centers = np.zeros(total_bins)
        sim_energy_bin_high = np.zeros(total_bins)
        sim_signal_events_per_bin = np.zeros_like(sim_energy_bin_centers)
        sim_differential_photon_fluxes = np.zeros_like(sim_energy_bin_centers)

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_sig_events = this_en_sig_th1d.values().sum()

        # Now let's see what has been simulated, i.e., the differential flux
        # at the center of each bin_name of the en_sig histogram
        bin_lower_edges = this_en_sig_th1d.axes.edges[0][:-1]
        bin_upper_edges = bin_lower_edges + this_en_sig_th1d.axes.widths[0]
        bin_centers = this_en_sig_th1d.axes.centers[0]
        bin_signal_events = this_en_sig_th1d.values()

        for i in range(this_en_sig_th1d.shape[0]):

            sim_energy_bin_low[i] = 10 ** bin_lower_edges[i]
            sim_energy_bin_centers[i] = 10 ** bin_centers[i]
            sim_energy_bin_high[i] = 10 ** bin_upper_edges[i]

            # NOTE: doesn't have the ability to read and evaluate TF1
            sim_differential_photon_fluxes[i] = 10 ** (
                log_log_spectrum(bin_centers[i], log_log_params)
            )

            sim_signal_events_per_bin[i] = bin_signal_events[i]

        # Read the histogram of the bkg events detected in this bin_name
        # Note: we do not copy this TH1D instance because we don't need it after the file is close
        sim_n_bg_events = this_en_bg_th1d.values().sum()

        # Now read the various TF1(s) for PSF, signal and background
        # Read the PSF and make a copy (so it will stay when we close the file)
        # NOTE: uproot doesn't have the ability to read and evaluate TF1
        try:

            psf_prefix = f"dec_{dec_id:02d}/nh_{analysis_bin_id}"
            psf_tf1_metadata = root_file[f"{psf_prefix}/PSF_dec{dec_id}_nh{analysis_bin_id}_fit"]
        except uproot.KeyInFileError:

            psf_prefix = f"dec_{dec_id:02d}/nh_0{analysis_bin_id}"
            psf_tf1_metadata = root_file[f"{psf_prefix}/PSF_dec{dec_id}_nh{analysis_bin_id}_fit"]

        psf_tf1_fparams = psf_tf1_metadata.member("fParams")

        psf_fun = PSFWrapper.psf_eval(psf_tf1_fparams)

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

    def to_pandas(self):
        """Save the information from Response file into a pandas.DataFrame

        Returns:
            tuple(pd.DataFrame): returns a tuple of pd.DataFrame, Response function metadata,
            and PSFWrapper instance
        """

        # In the metadata let's save all single values (floats)
        meta = {
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

    def combine_with_weights(self, other_response_bin, dec_center, w1, w2):
        """
        Produce another response bin which is the weighted sum of this one and the other one passed.

        :param other_response_bin:
        :param w1:
        :param w2:
        :return:
        """

        assert np.isclose(w1 + w2, 1.0), "Weights are not properly normalized"

        new_name = "interpolated_%s" % self._name

        # Use np.nan as declination boundaries to indicate that this is actually interpolated
        min_dec, max_dec = np.nan, np.nan

        n_sim_signal_events = (
            w1 * self._sim_n_sig_events + w2 * other_response_bin._sim_n_sig_events
        )
        n_sim_bkg_events = w1 * self._sim_n_bg_events + w2 * other_response_bin._sim_n_bg_events

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

        return new_response_bin

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
