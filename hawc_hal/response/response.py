import numpy as np
import pandas as pd
import os
import collections

from ..serialize import Serialization

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.exceptions.custom_exceptions import custom_warnings

from ..psf_fast import PSFWrapper
from response_bin import ResponseBin


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

        else:  # pragma: no cover

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

        response_bins = collections.OrderedDict()

        with Serialization(response_file_name, mode='r') as serializer:

            meta_dfs, _ = serializer.retrieve_pandas_object('/dec_bins_definition')
            effarea_dfs, _ = serializer.retrieve_pandas_object('/effective_area')
            psf_dfs, _ = serializer.retrieve_pandas_object('/psf')

        declination_centers = effarea_dfs.index.levels[0]
        energy_bins = effarea_dfs.index.levels[1]

        min_decs = []
        max_decs = []

        for dec_center in declination_centers:

            these_response_bins = collections.OrderedDict()

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

                sim_energy_bin_low = this_effarea_df.loc[:, 'sim_energy_bin_low'].values
                sim_energy_bin_centers = this_effarea_df.loc[:, 'sim_energy_bin_centers'].values
                sim_energy_bin_hi = this_effarea_df.loc[:, 'sim_energy_bin_hi'].values
                sim_differential_photon_fluxes = this_effarea_df.loc[:, 'sim_differential_photon_fluxes'].values
                sim_signal_events_per_bin = this_effarea_df.loc[:, 'sim_signal_events_per_bin'].values

                this_psf = PSFWrapper.from_pandas(psf_dfs.loc[dec_center, energy_bin])

                this_response_bin = ResponseBin(energy_bin, min_dec, max_dec, dec_center,
                                                sim_n_sig_events, sim_n_bg_events,
                                                sim_energy_bin_low,
                                                sim_energy_bin_centers,
                                                sim_energy_bin_hi,
                                                sim_differential_photon_fluxes,
                                                sim_signal_events_per_bin,
                                                this_psf)

                these_response_bins[energy_bin] = this_response_bin

            # Store the response bins for this declination bin

            response_bins[dec_center] = these_response_bins

        dec_bins = zip(min_decs, declination_centers, max_decs)

        return cls(response_file_name, dec_bins, response_bins)

    @classmethod
    def from_root_file(cls, response_file_name):

        from ..root_handler import open_ROOT_file, get_list_of_keys, tree_to_ndarray

        # Make sure file is readable

        response_file_name = sanitize_filename(response_file_name)

        # Check that they exists and can be read

        if not file_existing_and_readable(response_file_name):  # pragma: no cover
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
            try:

                response_bins_ids = tree_to_ndarray(f.Get("AnalysisBins"), "name")  # type: np.ndarray

            except ValueError:

                try:
                
                    response_bins_ids = tree_to_ndarray(f.Get("AnalysisBins"), "id")  # type: np.ndarray
                
                except ValueError:

                    # Some old response files (or energy responses) have no "name" branch
                    custom_warnings.warn("Response %s has no AnalysisBins 'id' or 'name' branch. "
                                     "Will try with default names" % response_file_name)

                    response_bins_ids = None
            response_bins_ids = response_bins_ids.astype(str)

            # Now we create a dictionary of ResponseBin instances for each dec bin_name
            response_bins = collections.OrderedDict()

            for dec_id in range(len(dec_bins)):

                this_response_bins = collections.OrderedDict()

                min_dec, dec_center, max_dec = dec_bins[dec_id]

                # If we couldn't get the reponse_bins_ids above, let's use the default names
                if response_bins_ids is None:

                    # Default are just integers. let's read how many nHit bins are from the first dec bin
                    dec_id_label = "dec_%02i" % dec_id

                    n_energy_bins = f.Get(dec_id_label).GetNkeys()

                    response_bins_ids = range(n_energy_bins)

                for response_bin_id in response_bins_ids:

                    this_response_bin = ResponseBin.from_ttree(f, dec_id, response_bin_id, log_log_spectrum,
                                                               min_dec, dec_center, max_dec)

                    this_response_bins[response_bin_id] = this_response_bin

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
        :param interpolate: whether to interpolate or not the PSF between the two closes response bins
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

            new_responses = collections.OrderedDict()

            for bin_id in energy_bins_one:

                this_new_response = energy_bins_one[bin_id].combine_with_weights(energy_bins_two[bin_id], dec, w1, w2)

                new_responses[bin_id] = this_new_response

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

    def display(self, verbose=False):
        """
        Prints summary of the current object content.

        :param verbose bool: Prints the full list of declinations and analysis bins.
        """

        print("Response file: %s" % self._response_file_name)
        print("Number of dec bins: %s" % len(self._dec_bins))
        if verbose:
            print self._dec_bins
        print("Number of energy/nHit planes per dec bin_name: %s" % (self.n_energy_planes))
        if verbose:
            print self._response_bins.values()[0].keys()

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

            for bin_id in self._response_bins[dec_center]:

                response_bin = self._response_bins[dec_center][bin_id]
                this_effarea_df, this_meta, this_psf_df = response_bin.to_pandas()

                effarea_dfs.append(this_effarea_df)
                psf_dfs.append(this_psf_df)
                assert bin_id == response_bin.name, \
                    'Bin name inconsistency: {} != {}'.format(bin_id, response_bin.name)
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
