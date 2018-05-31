import numpy as np
import pandas as pd
import os
from serialize import Serialization

try:

    import ROOT
    from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
    from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray

    ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)

except ImportError:

    pass

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from psf_fast import PSFWrapper


class ResponseBin(object):

    def __init__(self, name, min_dec, max_dec, dec_center,
                 sim_n_sig_events, sim_n_bg_events,
                 sim_energy_bin_centers, sim_differential_photon_fluxes, sim_signal_events_per_bin,
                 psf):
        self._name = name
        self._min_dec = min_dec
        self._max_dec = max_dec
        self._dec_center = dec_center
        self._sim_n_sig_events = sim_n_sig_events
        self._sim_n_bg_events = sim_n_bg_events
        self._sim_energy_bin_centers = sim_energy_bin_centers
        self._sim_differential_photon_fluxes = sim_differential_photon_fluxes
        self._sim_signal_events_per_bin = sim_signal_events_per_bin
        self._psf = psf  # type: PSFWrapper

    @classmethod
    def from_ttree(cls, open_ttree, dec_id, analysis_bin_id, log_log_spectrum, min_dec, dec_center, max_dec):

        # Compute the labels as used in the response file
        dec_id_label = "dec_%02i" % dec_id
        analysis_bin_id_label = "nh_%02i" % analysis_bin_id

        # Read the histogram of the simulated events detected in this bin_name
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        en_sig_label = "EnSig_dec%i_nh%i" % (dec_id, analysis_bin_id)

        # self._name = en_sig_label

        this_en_sig_th1d = open_ttree.Get("%s/%s/%s" % (dec_id_label, analysis_bin_id_label, en_sig_label))

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_sig_events = this_en_sig_th1d.Integral()

        # Get the content of the histogram as a numpy array
        # en_sig_hist = root_numpy.hist2array(this_en_sig_th1d,
        #                                           include_overflow=False,
        #                                           copy=True,
        #                                           return_edges=False)  # type: np.ndarray

        # Now let's see what has been simulated, i.e., the differential flux
        # at the center of each bin_name of the en_sig histogram
        sim_energy_bin_centers = np.zeros(this_en_sig_th1d.GetNbinsX())
        sim_signal_events_per_bin = np.zeros_like(sim_energy_bin_centers)
        sim_differential_photon_fluxes = np.zeros_like(sim_energy_bin_centers)

        for i in range(sim_energy_bin_centers.shape[0]):
            # Remember: bin_name 0 is the underflow bin_name, that is why there
            # is a "i+1" and not just "i"
            bin_center = this_en_sig_th1d.GetBinCenter(i + 1)

            # Store the center of the logarithmic bin_name
            sim_energy_bin_centers[i] = 10 ** bin_center  # TeV

            # Get from the simulated spectrum the value of the differential flux
            # at the center energy
            sim_differential_photon_fluxes[i] = 10 ** log_log_spectrum.Eval(bin_center)  # TeV^-1 cm^-1 s^-1

            # Get from the histogram the detected events in each log-energy bin_name
            sim_signal_events_per_bin[i] = this_en_sig_th1d.GetBinContent(i + 1)

        # Read the histogram of the bkg events detected in this bin_name
        # NOTE: we do not copy this TH1D instance because we won't use it after the
        # file is closed

        en_bg_label = "EnBg_dec%i_nh%i" % (dec_id, analysis_bin_id)
        this_en_bg_th1d = open_ttree.Get("%s/%s/%s" % (dec_id_label, analysis_bin_id_label, en_bg_label))

        # The sum of the histogram is the total number of simulated events detected
        # in this analysis bin_name
        sim_n_bg_events = this_en_bg_th1d.Integral()

        # Now read the various TF1(s) for PSF, signal and background

        # Read the PSF and make a copy (so it will stay when we close the file)

        psf_label_tf1 = "PSF_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        psf_fun = PSFWrapper.from_TF1(open_ttree.Get("%s/%s/%s" % (dec_id_label,
                                                                   analysis_bin_id_label,
                                                                   psf_label_tf1)))

        # en_sig_label_tf1 = "EnSig_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        # self._en_sig_fun = TF1Wrapper(open_ttree.Get("%s/%s/%s" % (dec_id_label,
        #                                                            analysis_bin_id_label,
        #                                                            en_sig_label_tf1)))
        #
        # en_bg_label_tf1 = "EnBg_dec%i_nh%i_fit" % (dec_id, analysis_bin_id)
        # self._en_bg_fun = TF1Wrapper(open_ttree.Get("%s/%s/%s" % (dec_id_label,
        #                                                           analysis_bin_id_label,
        #                                                           en_bg_label_tf1)))

        return cls(analysis_bin_id, min_dec, max_dec, dec_center, sim_n_sig_events, sim_n_bg_events,
                   sim_energy_bin_centers, sim_differential_photon_fluxes, sim_signal_events_per_bin,
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
            ('sim_energy_bin_centers', pd.Series(self.sim_energy_bin_centers)),
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

        # Now interpolate the psf
        new_psf = self._psf.combine_with_other_psf(other_response_bin._psf, w1, w2)

        new_response_bin = ResponseBin(new_name, min_dec, max_dec, dec_center,
                                       n_sim_signal_events, n_sim_bkg_events,
                                       self._sim_energy_bin_centers,
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
    def sim_energy_bin_centers(self):
        return self._sim_energy_bin_centers

    @property
    def sim_differential_photon_fluxes(self):
        return self._sim_differential_photon_fluxes

    @property
    def sim_signal_events_per_bin(self):
        return self._sim_signal_events_per_bin


_instances = {}


def hawc_response_factory(response_file_name):
    """
    A factory function for the response which keeps a cache, so that the same response is not read over and
    over again.

    :param response_file_name:
    :return: an instance of HAWCResponse
    """

    response_file_name = sanitize_filename(response_file_name, abspath=True)

    # See if this response is in the cache, if not build it

    if not response_file_name in _instances:

        print("Creating singleton for %s" % response_file_name)

        # Use the extension of the file to figure out which kind of response it is (ROOT or HDF)

        extension = os.path.splitext(response_file_name)[-1]

        if extension == ".root":

            new_instance = HAWCResponse.from_root_file(response_file_name)

        elif extension in ['.hd5', '.hdf5']:

            new_instance = HAWCResponse.from_hdf5(response_file_name)

        else:

            raise NotImplementedError("Extension %s for response file %s not recognized." % (extension,
                                                                                             response_file_name))

        _instances[response_file_name] = new_instance

    # return the response, whether it was already in the cache or we just built it

    return _instances[response_file_name]  # type: HAWCResponse


class HAWCResponse(object):

    def __init__(self, response_file_name, dec_bins, response_bins):

        self._response_file_name = response_file_name
        self._dec_bins = dec_bins
        self._response_bins = response_bins

    @classmethod
    def from_hdf5(cls, response_file_name):

        response_bins = {}

        with Serialization(response_file_name, mode='r') as serializer:

            meta_dfs, _ = serializer.retrieve_pandas_object('/dec_bins_definition')
            effarea_dfs, _ = serializer.retrieve_pandas_object('/effective_area')
            psf_dfs, _ = serializer.retrieve_pandas_object('/psf')

        declination_centers = effarea_dfs.index.levels[0]
        energy_bins = effarea_dfs.index.levels[1]

        min_decs = []
        max_decs = []

        for dec_center in declination_centers:

            these_response_bins = []

            for i, energy_bin in enumerate(energy_bins):

                these_meta = meta_dfs.loc[dec_center, energy_bin]

                min_dec = these_meta['min_dec']
                max_dec = these_meta['max_dec']
                dec_center_ = these_meta['declination_center']

                assert dec_center_ == dec_center, "Response is corrupted"

                # If this is the first energy bin, let's store the minimum and maximum dec of this bin
                if i == 0:

                    min_decs.append(min_dec)
                    max_decs.append(max_dec)

                else:

                    # Check that the minimum and maximum declination for this bin are the same as for
                    # the first energy bin
                    assert min_dec == min_decs[-1], "Response is corrupted"
                    assert max_dec == max_decs[-1], "Response is corrupted"

                sim_n_sig_events = these_meta['n_sim_signal_events']
                sim_n_bg_events = these_meta['n_sim_bkg_events']

                this_effarea_df = effarea_dfs.loc[dec_center, energy_bin]

                sim_energy_bin_centers = this_effarea_df.loc[:, 'sim_energy_bin_centers'].values
                sim_differential_photon_fluxes = this_effarea_df.loc[:, 'sim_differential_photon_fluxes'].values
                sim_signal_events_per_bin = this_effarea_df.loc[:, 'sim_signal_events_per_bin'].values

                this_psf = PSFWrapper.from_pandas(psf_dfs.loc[dec_center, energy_bin])

                this_response_bin = ResponseBin(energy_bin, min_dec, max_dec, dec_center,
                                                sim_n_sig_events, sim_n_bg_events,
                                                sim_energy_bin_centers, sim_differential_photon_fluxes,
                                                sim_signal_events_per_bin,
                                                this_psf)

                these_response_bins.append(this_response_bin)

            # Store the response bins for this declination bin

            response_bins[dec_center] = these_response_bins

        dec_bins = zip(min_decs, declination_centers, max_decs)

        return cls(response_file_name, dec_bins, response_bins)

    @classmethod
    def from_root_file(cls, response_file_name):

        # Make sure file is readable

        response_file_name = sanitize_filename(response_file_name)

        # Check that they exists and can be read

        if not file_existing_and_readable(response_file_name):
            raise IOError("Response %s does not exist or is not readable" % response_file_name)

        # Read response

        with open_ROOT_file(response_file_name) as f:

            # Get the name of the trees
            object_names = get_list_of_keys(f)

            # Make sure we have all the things we need

            assert 'LogLogSpectrum' in object_names
            assert 'DecBins' in object_names
            assert 'AnalysisBins' in object_names

            # Read spectrum used during the simulation
            log_log_spectrum = f.Get("LogLogSpectrum")

            # Get the analysis bins definition
            dec_bins_ = tree_to_ndarray(f.Get("DecBins"))

            dec_bins_lower_edge = dec_bins_['lowerEdge']  # type: np.ndarray
            dec_bins_upper_edge = dec_bins_['upperEdge']  # type: np.ndarray
            dec_bins_center = dec_bins_['simdec']  # type: np.ndarray

            dec_bins = zip(dec_bins_lower_edge, dec_bins_center, dec_bins_upper_edge)

            # Read in the ids of the response bins ("analysis bins" in LiFF jargon)
            response_bins_ids = tree_to_ndarray(f.Get("AnalysisBins"), "id")  # type: np.ndarray

            # Now we create a list of ResponseBin instances for each dec bin_name
            response_bins = {}

            for dec_id in range(len(dec_bins)):

                this_response_bins = []

                min_dec, dec_center, max_dec = dec_bins[dec_id]

                for response_bin_id in response_bins_ids:

                    this_response_bin = ResponseBin.from_ttree(f, dec_id, response_bin_id, log_log_spectrum,
                                                               min_dec, dec_center, max_dec)

                    this_response_bins.append(this_response_bin)

                response_bins[dec_bins[dec_id][1]] = this_response_bins

        # Now the file is closed. Let's explicitly remove f so we are sure it is freed
        del f

        # Instance the class and return it
        instance = cls(response_file_name, dec_bins, response_bins)

        return instance

    def get_response_dec_bin(self, dec, interpolate=False):
        """
        Get the responses for the provided declination bin, optionally interpolating the PSF

        :param dec: the declination where the response is desired at
        :param interpolate: whether to interpolate or not the PSF between the two closes reponse bins
        :return:
        """

        # Sort declination bins by distance to the provided declination
        dec_bins_keys = self._response_bins.keys()
        dec_bins_by_distance = sorted(dec_bins_keys, key=lambda x: abs(x - dec))

        if not interpolate:

            # Find the closest dec bin_name. We iterate over all the dec bins because we don't want to assume
            # that the bins are ordered by Dec in the file (and the operation is very cheap anyway,
            # since the dec bins are few)

            closest_dec_key = dec_bins_by_distance[0]

            return self._response_bins[closest_dec_key]

        else:

            # Find the two closest responses
            dec_bin_one, dec_bin_two = dec_bins_by_distance[:2]

            # Let's handle the special case where the requested dec is exactly on a response bin
            if abs(dec_bin_one - dec) < 0.01:

                # Do not interpolate
                return self._response_bins[dec_bin_one]

            energy_bins_one = self._response_bins[dec_bin_one]
            energy_bins_two = self._response_bins[dec_bin_two]

            # Now linearly interpolate between them

            # Compute the weights according to the distance to the source
            w1 = (dec - dec_bin_two) / (dec_bin_one - dec_bin_two)
            w2 = (dec - dec_bin_one) / (dec_bin_two - dec_bin_one)

            new_responses = []

            for i in range(len(energy_bins_one)):

                this_new_response = energy_bins_one[i].combine_with_weights(energy_bins_two[i], dec, w1, w2)

                new_responses.append(this_new_response)

            return new_responses


    @property
    def dec_bins(self):

        return self._dec_bins

    @property
    def response_bins(self):

        return self._response_bins

    @property
    def n_energy_planes(self):

        return len(self._response_bins.values()[0])

    def display(self):

        print("Response file: %s" % self._response_file_name)
        print("Number of dec bins: %s" % len(self._dec_bins))
        print("Number of energy/nHit planes per dec bin_name: %s" % (self.n_energy_planes))

    def write(self, filename):
        """
        Write the response to HDF5.

        :param filename: output file. WARNING: it will be overwritten if existing.
        :return:
        """

        filename = sanitize_filename(filename)

        # Unravel the dec bins
        min_decs, center_decs, max_decs = zip(*self._dec_bins)

        # We get the definition of the response bins, as well as their coordinates (the dec center) and store them
        # in lists. Later on we will use these to make 3 dataframes containing all the needed data
        multi_index_keys = []
        effarea_dfs = []
        psf_dfs = []
        all_metas = []

        # Loop over all the dec bins (making sure that they are in order)
        for dec_center in sorted(center_decs):

            for response_bin in self._response_bins[dec_center]:

                this_effarea_df, this_meta, this_psf_df = response_bin.to_pandas()

                effarea_dfs.append(this_effarea_df)
                psf_dfs.append(this_psf_df)
                multi_index_keys.append((dec_center, response_bin.name))
                all_metas.append(pd.Series(this_meta))

        # Create the dataframe with all the effective areas (with a multi-index)
        effarea_df = pd.concat(effarea_dfs, axis=0, keys=multi_index_keys)
        psf_df = pd.concat(psf_dfs, axis=0, keys=multi_index_keys)
        meta_df = pd.concat(all_metas, axis=1, keys=multi_index_keys).T

        # Now write the 4 dataframes to file
        with Serialization(filename, mode='w') as serializer:

            serializer.store_pandas_object('/dec_bins_definition', meta_df)
            serializer.store_pandas_object('/effective_area', effarea_df)
            serializer.store_pandas_object('/psf', psf_df)


