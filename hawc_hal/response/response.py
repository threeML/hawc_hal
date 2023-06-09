from __future__ import absolute_import, division

import collections
import os
from builtins import object, range, zip
from multiprocessing.managers import ValueProxy
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import uproot
from past.utils import old_div
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from ..serialize import Serialization

log = setup_logger(__name__)
log.propagate = False

from ..psf_fast import PSFWrapper
from .response_bin import ResponseBin

_instances = {}


def hawc_response_factory(response_file_name, bin_list2=None, dec_list2=None):
    """
    A factory function for the response which keeps a cache, so that the same response is not read over and
    over again.

    :param response_file_name:
    :return: an instance of HAWCResponse
    """

    response_file_name = sanitize_filename(response_file_name, abspath=True)

    # See if this response is in the cache, if not build it

    if not response_file_name in _instances:

        # log.info("Creating singleton for %s" % response_file_name)
        log.info(f"Creating singleton for {response_file_name}")

        # Use the extension of the file to figure out which kind of response it is (ROOT or HDF)

        extension = os.path.splitext(response_file_name)[-1]

        if extension == ".root":

            #new_instance = HAWCResponse.from_root_file(response_file_name)
            new_instance = HAWCResponse.from_root_file(response_file_name, bin_list2, dec_list2)
        elif extension in [".hd5", ".hdf5", ".hdf"]:

            new_instance = HAWCResponse.from_hdf5(response_file_name)

        else:  # pragma: no cover

            raise NotImplementedError(
                f"Extension {extension} for response file {response_file_name} not recognized."
            )
            # raise NotImplementedError("Extension %s for response file %s not recognized." % (extension,
            #  response_file_name))

        _instances[response_file_name] = new_instance

    # return the response, whether it was already in the cache or we just built it

    return _instances[response_file_name]  # type: HAWCResponse


class HAWCResponse(object):
    def __init__(self, response_file_name, dec_bins, response_bins):

        self._response_file_name = response_file_name
        self._dec_bins = dec_bins
        self._response_bins = response_bins

        if len(dec_bins) < 2:
            #   log.warning("Only {0} dec bins given in {1}, will not try to interpolate.".format(len(dec_bins), response_file_name))
            log.warning(
                f"Only {len(dec_bins)} dec bins given in {response_file_name}, will not try to interpolate."
            )
            log.warning(
                "Single-dec-bin mode is intended for development work only at this time and may not work with extended sources."
            )

    @classmethod
    def from_hdf5(cls, response_file_name):
        """
        Build response from a HDF5 file. Do not use directly,
        use the hawc_response_factory function instead.

        :param response_file_name:
        :return: a HAWCResponse instance
        """

        response_bins = collections.OrderedDict()

        with Serialization(response_file_name, mode="r") as serializer:

            meta_dfs, _ = serializer.retrieve_pandas_object("/dec_bins_definition")
            effarea_dfs, _ = serializer.retrieve_pandas_object("/effective_area")
            psf_dfs, _ = serializer.retrieve_pandas_object("/psf")

        declination_centers = effarea_dfs.index.levels[0]
        energy_bins = effarea_dfs.index.levels[1]

        min_decs = []
        max_decs = []

        for dec_center in declination_centers:

            these_response_bins = collections.OrderedDict()

            for i, energy_bin in enumerate(energy_bins):

                these_meta = meta_dfs.loc[dec_center, energy_bin]

                min_dec = these_meta["min_dec"]
                max_dec = these_meta["max_dec"]
                dec_center_ = these_meta["declination_center"]

                assert dec_center_ == dec_center, "Response is corrupted"

                # If this is the first energy bin, let's store the minimum and
                # maximum dec of this bin
                if i == 0:

                    min_decs.append(min_dec)
                    max_decs.append(max_dec)

                else:

                    # Check that the minimum and maximum declination for this bin are
                    # the same as for the first energy bin
                    assert min_dec == min_decs[-1], "Response is corrupted"
                    assert max_dec == max_decs[-1], "Response is corrupted"

                sim_n_sig_events = these_meta["n_sim_signal_events"]
                sim_n_bg_events = these_meta["n_sim_bkg_events"]

                this_effarea_df = effarea_dfs.loc[dec_center, energy_bin]

                sim_energy_bin_low = this_effarea_df.loc[:, "sim_energy_bin_low"].values
                sim_energy_bin_centers = this_effarea_df.loc[:, "sim_energy_bin_centers"].values
                sim_energy_bin_hi = this_effarea_df.loc[:, "sim_energy_bin_hi"].values
                sim_differential_photon_fluxes = this_effarea_df.loc[
                    :, "sim_differential_photon_fluxes"
                ].values
                sim_signal_events_per_bin = this_effarea_df.loc[
                    :, "sim_signal_events_per_bin"
                ].values

                this_psf = PSFWrapper.from_pandas(psf_dfs.loc[dec_center, energy_bin, :])

                this_response_bin = ResponseBin(
                    energy_bin,
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
                    this_psf,
                )

                these_response_bins[energy_bin] = this_response_bin

            # Store the response bins for this declination bin

            response_bins[dec_center] = these_response_bins

        dec_bins = list(zip(min_decs, declination_centers, max_decs))

        return cls(response_file_name, dec_bins, response_bins)

    @classmethod
    def from_root_file(cls, response_file_name: Path, bin_list2, dec_list2):
        """Build response from ROOT file. Do not use directly, use the
        hawc_response_factory instead.

        Args:
            response_file_name (str): name of response file name

        Raises:
            IOError: An IOError is raised if the file is corrupted or unreadable

        Returns:
            HAWCResponse: returns a HAWCResponse instance
        """

        # from ..root_handler import open_ROOT_file, get_list_of_keys, tree_to_ndarray

        # Make sure file is readable

        response_file_name = sanitize_filename(response_file_name)

        # Check that they exists and can be read

        if not file_existing_and_readable(response_file_name):  # pragma: no cover

            # raise IOError("Response %s does not exist or is not readable" % response_file_name)
            raise IOError(f"Response {response_file_name} does not exist or is not readable")

        # Read response
        with uproot.open(str(response_file_name)) as response_file:

            log.info("Reading Response File!")
            # NOTE: The LogLogSpectrum parameters are extracted from the response file
            # There is no way to evaluate the loglogspectrum function with uproot, so the
            # parameters are passed for evaluation in response_bin.py

            log_log_params = response_file["LogLogSpectrum"].member("fParams")
            log_log_shape = response_file["LogLogSpectrum"].member("fTitle")
            dec_bins_lower_edge = response_file["DecBins/lowerEdge"].array().to_numpy()
            dec_bins_upper_edge = response_file["DecBins/upperEdge"].array().to_numpy()
            dec_bins_center = response_file["DecBins/simdec"].array().to_numpy()

            dec_bins = list(zip(dec_bins_lower_edge, dec_bins_center, dec_bins_upper_edge))

            try:

                response_bin_ids = response_file["AnalysisBins/name"].array().to_numpy()

            except uproot.KeyInFileError:

                try:

                    response_bin_ids = response_file["AnalysisBins/id"].array().to_numpy()

                except uproot.KeyInFileError:

                    log.warning(
                        f"Response {response_file_name} has no AnalysisBins 'id'"
                        "or 'name' branch. Will try with the default names"
                    )

                    response_bin_ids = None

            response_bin_ids = response_bin_ids.astype(str)

            # Now we create a dictionary of ResponseBin instances for each dec bin name
            response_bins = collections.OrderedDict()

            #for dec_id in range(len(dec_bins)):
            for dec_id in dec_list2:

                this_response_bins = collections.OrderedDict()
                min_dec, dec_center, max_dec = dec_bins[dec_id]

                if response_bin_ids is None:

                    dec_id_label = f"dec_{dec_id:02d}"

                    # count the number of keys if name of bins wasn't obtained before
                    n_energy_bins = len(response_file[dec_id_label].keys(recursive=False))

                    response_bins_ids = list(range(n_energy_bins))
                log.info("Dec ID= %s" %(dec_id))
                for response_bin_id in response_bin_ids:

                    this_response_bin = ResponseBin.from_ttree(
                        response_file,
                        dec_id,
                        response_bin_id,
                        log_log_params,
                        log_log_shape,
                        min_dec,
                        dec_center,
                        max_dec, 
                        bin_list2
                    )

                    this_response_bins[response_bin_id] = this_response_bin

                response_bins[dec_bins[dec_id][1]] = this_response_bins

            del response_file

        return cls(response_file_name, dec_bins, response_bins)

    def get_response_dec_bin(self, dec, interpolate=False):
        """Get the response for the provided declination bin, optionally interpolating the PSF


        Args:
            dec (float): Declination where the response is desired
            interpolate (bool, optional): If True, PSF is interpolated between the two
            closest response bins. Defaults to False.

        Returns:
            PSFWrapper: To be added later.
        """

        # Sort declination bins by distance to the provided declination
        dec_bins_keys = list(self._response_bins.keys())
        dec_bins_by_distance = sorted(dec_bins_keys, key=lambda x: abs(x - dec))

        # never try to interpolate if only one dec bin is given
        if len(dec_bins_keys) < 2:
            interpolate = False

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
            w1 = old_div((dec - dec_bin_two), (dec_bin_one - dec_bin_two))
            w2 = old_div((dec - dec_bin_one), (dec_bin_two - dec_bin_one))

            new_responses = collections.OrderedDict()

            for bin_id in energy_bins_one:
                this_new_response = energy_bins_one[bin_id].combine_with_weights(
                    energy_bins_two[bin_id], dec, w1, w2
                )

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

        return len(list(self._response_bins.values())[0])

    def display(self, verbose=False):
        """
        Prints summary of the current object content.

        :param verbose bool: Prints the full list of declinations and analysis bins.
        """

        # log.info("Response file: %s" % self._response_file_name)
        # log.info("Number of dec bins: %s" % len(self._dec_bins))
        log.info(f"Response file: {self._response_file_name}")
        log.info(f"Number of dec bins: {len(self._dec_bins)}")
        if verbose:
            log.info(self._dec_bins)
        # log.info("Number of energy/nHit planes per dec bin_name: %s" % (self.n_energy_planes))
        log.info(f"Number of energy/nHit planes per dec bin_name: {self.n_energy_planes}")
        if verbose:
            log.info(list(self._response_bins.values())[0].keys())

    def write(self, filename):
        """
        Write the response to HDF5 file.

        Args:
            filename (str): Output file name. WARNING: it will be overwritten if
            file already exists.
        """

        filename = sanitize_filename(filename)

        # Unravel the dec bins
        min_decs, center_decs, max_decs = list(zip(*self._dec_bins))

        # We get the definition of the response bins, as well as their coordinates
        #  (the dec center) and store them
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
                # assert bin_id == response_bin.name, \
                # 'Bin name inconsistency: {} != {}'.format(bin_id, response_bin.name)
                assert (
                    bin_id == response_bin.name
                ), f"Bin name inconsistency: {bin_id} != {response_bin.name}"
                multi_index_keys.append((dec_center, response_bin.name))
                all_metas.append(pd.Series(this_meta))

        # Create the dataframe with all the effective areas (with a multi-index)
        effarea_df = pd.concat(effarea_dfs, axis=0, keys=multi_index_keys)
        psf_df = pd.concat(psf_dfs, axis=0, keys=multi_index_keys)
        meta_df = pd.concat(all_metas, axis=1, keys=multi_index_keys).T

        # Now write the 4 dataframes to file
        with Serialization(filename, mode="w") as serializer:

            serializer.store_pandas_object("/dec_bins_definition", meta_df)
            serializer.store_pandas_object("/effective_area", effarea_df)
            serializer.store_pandas_object("/psf", psf_df)
