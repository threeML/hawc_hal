from __future__ import absolute_import, division

import collections
import os
from builtins import object, zip
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from past.utils import old_div
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from ..psf_fast import PSFWrapper
from ..serialize import Serialization
from .response_bin import ResponseBin

log = setup_logger(__name__)
log.propagate = False
_instances = {}


def hawc_response_factory(response_file_name, n_workers: int):
    """
    A factory function for the response which keeps a cache, so that the same response is not read over and
    over again.

    :param response_file_name:
    :return: an instance of HAWCResponse
    """

    response_file_name = sanitize_filename(response_file_name, abspath=True)

    # See if this response is in the cache, if not build it

    if response_file_name not in _instances:
        # log.info("Creating singleton for %s" % response_file_name)
        log.info(f"Creating singleton for {response_file_name}")

        # Use the extension of the file to figure out which kind of response it is (ROOT or HDF)

        extension = os.path.splitext(response_file_name)[-1]

        if extension == ".root":
            new_instance = HAWCResponse.from_root_file(response_file_name, n_workers)

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


@dataclass
class ResponseBinMetaData:
    response_ttree_directory: uproot.ReadOnlyDirectory

    def generate_metadata(self, dec_id):
        psf_meta_dict = {}
        energy_meta_dict = {}
        energy_bkg_meta_dict = {}

        energy_meta_dict[dec_id] = [
            self.response_ttree_directory[
                f"dec_{dec_id:02d}/nh_{bin_id}/EnSig_dec{dec_id}_nh{bin_id}"
            ].to_hist()
            for bin_id in self.analysis_bins
        ]

        energy_bkg_meta_dict[dec_id] = [
            self.response_ttree_directory[
                f"dec_{dec_id:02d}/nh_{bin_id}/EnBg_dec{dec_id}_nh{bin_id}"
            ].to_hist()
            for bin_id in self.analysis_bins
        ]

        psf_meta_dict[dec_id] = [
            self.response_ttree_directory[
                f"dec_{dec_id:02d}/nh_{bin_id}/PSF_dec{dec_id}_nh{bin_id}_fit"
            ].member("fParams")
            for bin_id in self.analysis_bins
        ]

        return psf_meta_dict, energy_meta_dict, energy_bkg_meta_dict

    @property
    def declination_bins_lower(self) -> np.ndarray:
        if self.response_ttree_directory.get("DecBins/lowerEdge", None) is not None:
            return self.response_ttree_directory["DecBins/lowerEdge"].array(
                library="np"
            )
        else:
            raise KeyError("DecBins/lowerEdge not found in ROOT file")

    @property
    def declination_bins_upper(self) -> np.ndarray:
        if self.response_ttree_directory.get("DecBins/upperEdge", None) is not None:
            return self.response_ttree_directory["DecBins/upperEdge"].array(
                library="np"
            )
        else:
            raise KeyError("DecBins/upperEdge not found in ROOT file")

    @property
    def declination_bins_center(self) -> np.ndarray:
        if self.response_ttree_directory.get("DecBins/simdec", None) is not None:
            return self.response_ttree_directory["DecBins/simdec"].array(library="np")
        else:
            raise KeyError("DecBins/simdec not found in ROOT file")

    @property
    def analysis_bins(self) -> np.ndarray:
        try:
            return self.response_ttree_directory["AnalysisBins/name"].array(
                library="np"
            )
        except KeyError:
            log.warning(
                "AnalysisBins/name not found in ROOT file. Trying AnalysisBins/id"
            )
            return self.response_ttree_directory["AnalysisBins/id"].array(library="np")

    @property
    def log_log_params(self) -> np.ndarray:
        """Read the best-fit params from PSF fit from ROOT file"""
        if self.response_ttree_directory.get("LogLogSpectrum", None) is not None:
            return np.array(
                self.response_ttree_directory["LogLogSpectrum"].member("fParams")
            )
        else:
            raise KeyError("LogLogSpectrum not found in ROOT file")

    @property
    def spectrum_shape(self) -> str:
        """Read the best-fit params from PSF fit from ROOT file"""
        if self.response_ttree_directory.get("LogLogSpectrum", None) is not None:
            return self.response_ttree_directory["LogLogSpectrum"].member("fTitle")
        else:
            raise KeyError("LogLogSpectrum not found in ROOT file")


# ? Might or might not want to keep this, but will leave it here for now
class ResponseBinParams:
    def __init__(
        self,
        response_file_dir,
        dec_id,
        response_bin_id,
        log_log_params,
        log_log_shape,
        min_dec,
        dec_center,
        max_dec,
    ):
        self.response_file_dir = response_file_dir
        self.dec_id = dec_id
        self.response_bin_id = response_bin_id
        self.log_log_params = log_log_params
        self.log_log_shape = log_log_shape
        self.min_dec = min_dec
        self.dec_center = dec_center
        self.max_dec = max_dec


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
                sim_energy_bin_centers = this_effarea_df.loc[
                    :, "sim_energy_bin_centers"
                ].values
                sim_energy_bin_hi = this_effarea_df.loc[:, "sim_energy_bin_hi"].values
                sim_differential_photon_fluxes = this_effarea_df.loc[
                    :, "sim_differential_photon_fluxes"
                ].values
                sim_signal_events_per_bin = this_effarea_df.loc[
                    :, "sim_signal_events_per_bin"
                ].values

                this_psf = PSFWrapper.from_pandas(
                    psf_dfs.loc[dec_center, energy_bin, :]
                )

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
    def from_root_file(cls, response_file_name: Path, n_workers: int):
        """Build response from ROOT file. Do not use directly, use the
        hawc_response_factory instead.

        Args:
            response_file_name (str): name of response file name

        Raises:
            IOError: An IOError is raised if the file is corrupted or unreadable

        Returns:
            HAWCResponse: returns a HAWCResponse instance
        """

        # Make sure file is readable

        response_file_name = sanitize_filename(response_file_name)

        # Check that they exists and can be read

        if not file_existing_and_readable(response_file_name):  # pragma: no cover
            # raise IOError("Response %s does not exist or is not readable" % response_file_name)
            raise IOError(
                f"Response {response_file_name} does not exist or is not readable"
            )

        with uproot.open(response_file_name) as response_file_directory:
            resp_metadata = ResponseBinMetaData(response_file_directory)

            # NOTE:Get the Response function basic information
            log_log_params = resp_metadata.log_log_params
            log_log_shape = resp_metadata.spectrum_shape
            analysis_bins_arr = resp_metadata.analysis_bins
            dec_bins_lower_edge = resp_metadata.declination_bins_lower
            dec_bins_upper_edge = resp_metadata.declination_bins_upper
            dec_bins_sim = resp_metadata.declination_bins_center

            dec_bins = list(zip(dec_bins_lower_edge, dec_bins_sim, dec_bins_upper_edge))
            number_of_dec_bins = len(dec_bins_sim)

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(
                    resp_metadata.generate_metadata, list(range(number_of_dec_bins))
                )

            psf_meta_dict = {}
            energy_meta_dict = {}
            energy_bkg_meta_dict = {}

            for psf_meta, energy_meta, energy_bkg_meta in results:
                psf_meta_dict.update(psf_meta)
                energy_meta_dict.update(energy_meta)
                energy_bkg_meta_dict.update(energy_bkg_meta)

        # NOTE: Now we have all the info we need to build the response
        # TODO: refactor the code below and make more concise
        response_bins = collections.OrderedDict()
        for dec_id, dec_bin in enumerate(dec_bins):
            min_dec, dec_center, max_dec = dec_bin
            response_bins_per_dec = collections.OrderedDict()

            current_hist_array = energy_meta_dict[dec_id]
            current_hist_bkg_array = energy_bkg_meta_dict[dec_id]
            psf_vals_array = psf_meta_dict[dec_id]

            for bin_idx, bin_id in enumerate(analysis_bins_arr):
                current_hist = current_hist_array[bin_idx]
                current_hist_bkg = current_hist_bkg_array[bin_idx]
                current_psf_params = psf_vals_array[bin_idx]

                this_response_bin = ResponseBin.from_ttree(
                    bin_id,
                    current_hist,
                    current_hist_bkg,
                    current_psf_params,
                    log_log_params,
                    log_log_shape,
                    min_dec,
                    dec_center,
                    max_dec,
                )
                response_bins_per_dec[bin_id] = this_response_bin
            response_bins[dec_center] = response_bins_per_dec

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
        log.info(
            f"Number of energy/nHit planes per dec bin_name: {self.n_energy_planes}"
        )
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
            serializer.store_pandas_object("/psf", psf_df)
