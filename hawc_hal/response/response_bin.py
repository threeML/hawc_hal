import numpy as np
import pandas as pd

from threeML.exceptions.custom_exceptions import custom_warnings

from ..psf_fast import PSFWrapper, InvalidPSF, InvalidPSFError


class ResponseBin(object):
    """
    Stores detector response for one declination band and one analysis bin (called "name" or "analysis_bin_id" below).
    """

    def __init__(self, name, min_dec, max_dec, dec_center,
                 sim_n_sig_events, sim_n_bg_events,
                 sim_energy_bin_low, sim_energy_bin_centers, sim_energy_bin_hi,
                 sim_differential_photon_fluxes, sim_signal_events_per_bin,
                 psf):

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
    def _get_en_th1d(open_ttree, dec_id, analysis_bin_id, prefix):

        en_sig_label = "En%s_dec%i_nh%s" % (prefix, dec_id, analysis_bin_id)

        this_en_th1d = open_ttree.FindObjectAny(en_sig_label) 

        if not this_en_th1d:

            raise IOError("Could not find TH1D named %s." % en_sig_label)

        return this_en_th1d

    @classmethod
    def from_ttree(cls, open_ttree, dec_id, analysis_bin_id, log_log_spectrum, min_dec, dec_center, max_dec):

        from ..root_handler import ROOT

        # Read the histogram of the simulated events detected in this bin_name
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        this_en_sig_th1d = cls._get_en_th1d(open_ttree, dec_id, analysis_bin_id, 'Sig')

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_sig_events = this_en_sig_th1d.Integral()

        # Now let's see what has been simulated, i.e., the differential flux
        # at the center of each bin_name of the en_sig histogram
        sim_energy_bin_low = np.zeros(this_en_sig_th1d.GetNbinsX())
        sim_energy_bin_centers = np.zeros(this_en_sig_th1d.GetNbinsX())
        sim_energy_bin_hi = np.zeros(this_en_sig_th1d.GetNbinsX())
        sim_signal_events_per_bin = np.zeros_like(sim_energy_bin_centers)
        sim_differential_photon_fluxes = np.zeros_like(sim_energy_bin_centers)

        for i in range(sim_energy_bin_centers.shape[0]):
            # Remember: bin_name 0 is the underflow bin_name, that is why there
            # is a "i+1" and not just "i"
            bin_lo = this_en_sig_th1d.GetBinLowEdge(i+1)
            bin_center = this_en_sig_th1d.GetBinCenter(i + 1)
            bin_hi = this_en_sig_th1d.GetBinWidth(i+1) + bin_lo

            # Store the center of the logarithmic bin_name
            sim_energy_bin_low[i] = 10 ** bin_lo  # TeV
            sim_energy_bin_centers[i] = 10 ** bin_center  # TeV
            sim_energy_bin_hi[i] = 10 ** bin_hi  # TeV

            # Get from the simulated spectrum the value of the differential flux
            # at the center energy
            sim_differential_photon_fluxes[i] = 10 ** log_log_spectrum.Eval(bin_center)  # TeV^-1 cm^-1 s^-1

            # Get from the histogram the detected events in each log-energy bin_name
            sim_signal_events_per_bin[i] = this_en_sig_th1d.GetBinContent(i + 1)

        # Read the histogram of the bkg events detected in this bin_name
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        this_en_bg_th1d = cls._get_en_th1d(open_ttree, dec_id, analysis_bin_id, 'Bg')

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_bg_events = this_en_bg_th1d.Integral()

        # Now read the various TF1(s) for PSF, signal and background

        # Read the PSF and make a copy (so it will stay when we close the file)

        psf_label_tf1 = "PSF_dec%i_nh%s_fit" % (dec_id, analysis_bin_id)

        tf1 = open_ttree.FindObjectAny(psf_label_tf1)

        psf_fun = PSFWrapper.from_TF1(tf1)

        return cls(analysis_bin_id, min_dec, max_dec, dec_center, sim_n_sig_events, sim_n_bg_events,
                   sim_energy_bin_low,
                   sim_energy_bin_centers,
                   sim_energy_bin_hi,
                   sim_differential_photon_fluxes, sim_signal_events_per_bin,
                   psf_fun)

    def to_pandas(self):

        # In the metadata let's save all single values (floats)
        meta = {'min_dec': self._min_dec,
                'max_dec': self._max_dec,
                'declination_center': self._dec_center,
                'n_sim_signal_events': self._sim_n_sig_events,
                'n_sim_bkg_events': self._sim_n_bg_events
                }

        # Now make a dataframe containing the elements of the simulation
        items = (
            ('sim_energy_bin_low', pd.Series(self.sim_energy_bin_low)),
            ('sim_energy_bin_centers', pd.Series(self.sim_energy_bin_centers)),
            ('sim_energy_bin_hi', pd.Series(self.sim_energy_bin_hi)),
            ('sim_differential_photon_fluxes', pd.Series(self.sim_differential_photon_fluxes)),
            ('sim_signal_events_per_bin', pd.Series(self.sim_signal_events_per_bin))
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

        assert np.isclose(w1+w2, 1.0), "Weights are not properly normalized"

        new_name = "interpolated_%s" % self._name

        # Use np.nan as declination boundaries to indicate that this is actually interpolated
        min_dec, max_dec = np.nan, np.nan

        n_sim_signal_events = w1 * self._sim_n_sig_events + w2 * other_response_bin._sim_n_sig_events
        n_sim_bkg_events = w1 * self._sim_n_bg_events + w2 * other_response_bin._sim_n_bg_events

        # We assume that the bin centers are the same
        assert np.allclose(self._sim_energy_bin_centers, other_response_bin._sim_energy_bin_centers)

        sim_differential_photon_fluxes = w1 * self._sim_differential_photon_fluxes + \
                                         w2 * other_response_bin._sim_differential_photon_fluxes

        sim_signal_events_per_bin = w1 * self._sim_signal_events_per_bin + \
                                    w2 * other_response_bin._sim_signal_events_per_bin

        # Now interpolate the psf, if none is invalid
        try:
            new_psf = self._psf.combine_with_other_psf(other_response_bin._psf, w1, w2)
        except InvalidPSFError:
            new_psf = InvalidPSF()

        new_response_bin = ResponseBin(new_name, min_dec, max_dec, dec_center,
                                       n_sim_signal_events, n_sim_bkg_events,
                                       self._sim_energy_bin_low,
                                       self._sim_energy_bin_centers,
                                       self._sim_energy_bin_hi,
                                       sim_differential_photon_fluxes,
                                       sim_signal_events_per_bin,
                                       new_psf)

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
