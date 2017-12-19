import numpy as np
import collections
import ROOT
import root_numpy

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tgraph_to_arrays, th2_to_arrays, tree_to_ndarray


class TF1Wrapper(object):

    def __init__(self, tf1_instance):

        # Make a copy so that if the passed instance was a pointer from a TFile,
        # it will survive the closing of the associated TFile

        self._tf1 = ROOT.TF1(tf1_instance)

    def integral(self, *args, **kwargs):

        return self._tf1.Integral(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        return self._tf1.Eval(*args, **kwargs)


class AnalysisBin(object):

    def __init__(self, open_ttree, dec_id, analysis_bin_id, log_log_spectrum):

        # Compute the labels as used in the response file
        dec_id_label = "dec_%02i" % dec_id
        analysis_bin_id_label = "nh_%02i" % analysis_bin_id

        # Read the histogram of the simulated events detected in this bin
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        en_sig_label = "EnSig_dec%i_nh%i" % (dec_id, analysis_bin_id)
        this_en_sig_th1d = open_ttree.Get("%s/%s/%s" % (dec_id_label, analysis_bin_id_label, en_sig_label))

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin
        self._sim_n_sig_events = this_en_sig_th1d.Integral()

        # Get the content of the histogram as a numpy array
        self._en_sig_hist = root_numpy.hist2array(this_en_sig_th1d,
                                                  include_overflow=False,
                                                  copy=True,
                                                  return_edges=False)  # type: np.ndarray

        # Now let's see what has been simulated, i.e., the differential flux
        # at the center of each bin of the en_sig histogram
        self._en_sig_log_energy = np.zeros(this_en_sig_th1d.GetNbinsX())
        self._en_sig_simulated_diff_flux = np.zeros_like(self._en_sig_log_energy)

        for i in range(self._en_sig_log_energy.shape[0]):
            # Remember: bin 0 is the underflow bin, that is why there
            # is a "i+1" and not just "i"
            bin_center = this_en_sig_th1d.GetBinCenter(i + 1)

            # Store the center of the logarithmic bin
            self._en_sig_log_energy[i] = bin_center

            # Get from the simulated spectrum the value of the differential flux
            # at the center energy
            self._en_sig_simulated_diff_flux[i] = 10 ** log_log_spectrum(bin_center)

        # Read the histogram of the bkg events detected in this bin
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        en_bg_label = "EnBg_dec%i_nh%i" % (dec_id, analysis_bin_id)
        this_en_bg_th1d = open_ttree.Get("%s/%s/%s" % (dec_id_label, analysis_bin_id_label, en_bg_label))

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin
        self._sim_n_bg_events = this_en_bg_th1d.Integral()

        # Now read the various TF1(s) for PSF, signal and background

        # Read the PSF and make a copy (so it will stay when we close the file)

        psf_label_tf1 = "PSF_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        self._psf_fun = TF1Wrapper(open_ttree.Get("%s/%s/%s" % (dec_id_label,
                                                                analysis_bin_id_label,
                                                                psf_label_tf1)))

        en_sig_label_tf1 = "EnSig_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        self._en_sig_fun = TF1Wrapper(open_ttree.Get("%s/%s/%s" % (dec_id_label,
                                                                   analysis_bin_id_label,
                                                                   en_sig_label_tf1)))

        en_bg_label_tf1 = "EnBg_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        self._en_bg_fun = TF1Wrapper(open_ttree.Get("%s/%s/%s" % (dec_id_label,
                                                                  analysis_bin_id_label,
                                                                  en_bg_label_tf1)))


    @property
    def psf(self):

        return self._psf_fun

    @property
    def n_sim_signal_events(self):

        return self._sim_n_sig_events

    @property
    def n_sim_bkg_events(self):

        return self._sim_n_bg_events


class ResponseFile(object):

    def __init__(self, response_file_name):

        with open_ROOT_file(response_file_name) as f:

            # Get the name of the trees
            object_names = get_list_of_keys(f)

            # Make sure we have all the things we need

            assert 'LogLogSpectrum' in object_names
            assert 'DecBins' in object_names
            assert 'AnalysisBins' in object_names

            # Read spectrum used during the simulation
            self._log_log_spectrum = TF1Wrapper(ROOT.TF1(f.Get("LogLogSpectrum")))

            # Get the analysis bins definition
            dec_bins = tree_to_ndarray(f.Get("DecBins"))

            dec_bins_lower_edge = dec_bins['lowerEdge']  # type: np.ndarray
            dec_bins_upper_edge = dec_bins['upperEdge']  # type: np.ndarray
            dec_bins_center = dec_bins['simdec']  # type: np.ndarray

            self._dec_bins = zip(dec_bins_lower_edge, dec_bins_center, dec_bins_upper_edge)

            # Read in the ids of the analysis bins
            analysis_bins_ids = tree_to_ndarray(f.Get("AnalysisBins"), "id")  # type: np.ndarray

            # Now we create a list of AnalysisBin instances for each dec bin
            self._analysis_bins = collections.OrderedDict()

            for dec_id in range(len(self._dec_bins)):

                this_analysis_bins = []

                for analysis_bin_id in analysis_bins_ids:

                    this_analysis_bin = AnalysisBin(f, dec_id, analysis_bin_id, self._log_log_spectrum)

                    this_analysis_bins.append(this_analysis_bin)

                self._analysis_bins[dec_id] = this_analysis_bins

    @property
    def dec_bins(self):

        return self._dec_bins

    @property
    def analysis_bins(self):

        return self._analysis_bins