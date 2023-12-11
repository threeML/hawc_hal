from __future__ import absolute_import, division

import collections
import multiprocessing
import os
from builtins import object, zip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from numpy.typing import NDArray
from past.utils import old_div
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from ..psf_fast import PSFWrapper
from ..serialize import Serialization
from .response_bin import ResponseBin

log = setup_logger(__name__)
log.propagate = False
_instances = {}


def hawc_response_factory(response_file_name, n_workers: int = 1):
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
class ResponseMetaData:
    """Class to read the response metadata from a ROOT response file"""

    response_ttree_directory: uproot.ReadOnlyDirectory

    @staticmethod
    def get_energy_hist(
        response_ttree_directory: uproot.ReadOnlyDirectory, dec_id: int, bin_id: str
    ):
        """Read in the signal energy histogram from response file"""

        # dec_id, bin_id = args

        energy_hist_prefix = (
            f"dec_{dec_id:02d}/nh_{bin_id}/EnSig_dec{dec_id}_nh{bin_id}"
        )
        if response_ttree_directory.get(energy_hist_prefix, None) is not None:
            energy_hist = response_ttree_directory[energy_hist_prefix]

            return dec_id, bin_id, energy_hist.to_boost()  # type: ignore

        energy_hist_prefix = (
            f"dec_{dec_id:02d}/nh_{bin_id.zfill(2)}/EnSig_dec{dec_id}_nh{bin_id}"
        )

        if response_ttree_directory.get(energy_hist_prefix, None) is not None:
            energy_hist = response_ttree_directory[energy_hist_prefix]

            return dec_id, bin_id, energy_hist.to_boost()  # type: ignore

        raise KeyError("Unknown binning scheme in response file")

    @staticmethod
    def get_energy_bkg_hist(
        response_ttree_directory: uproot.ReadOnlyDirectory, dec_id: int, bin_id: str
    ):
        """Read in the background energy histogram from response file"""
        # dec_id, bin_id = args

        energy_bkg_prefix = f"dec_{dec_id:02d}/nh_{bin_id}/EnBg_dec{dec_id}_nh{bin_id}"

        if response_ttree_directory.get(energy_bkg_prefix, None) is not None:
            energy_bkg_hist = response_ttree_directory[energy_bkg_prefix]
            return dec_id, bin_id, energy_bkg_hist.to_boost()  # type: ignore

        energy_bkg_prefix = (
            f"dec_{dec_id:02d}/nh_{bin_id.zfill(2)}/EnBg_dec{dec_id}_nh{bin_id}"
        )

        if response_ttree_directory.get(energy_bkg_prefix, None) is not None:
            energy_bkg_hist = response_ttree_directory[energy_bkg_prefix]

            return dec_id, bin_id, energy_bkg_hist.to_boost()  # type: ignore

        raise KeyError("Unknown binning scheme in response file")

    @staticmethod
    def get_psf_params(
        response_ttree_directory: uproot.ReadOnlyDirectory, dec_id: int, bin_id: str
    ):
        """Read the list of best-fit params from PSF fit from response file"""
        # dec_id, bin_id = args
        psf_prefix = f"dec_{dec_id:02d}/nh_{bin_id}/PSF_dec{dec_id}_nh{bin_id}_fit"

        if response_ttree_directory.get(psf_prefix, None) is not None:
            psf_meta = response_ttree_directory[psf_prefix]
            return dec_id, bin_id, psf_meta.member("fParams")  # type: ignore

        psf_prefix = (
            f"dec_{dec_id:02d}/nh_{bin_id.zfill(2)}/PSF_dec{dec_id}_nh{bin_id}_fit"
        )
        if response_ttree_directory.get(psf_prefix, None) is not None:
            psf_meta = response_ttree_directory[psf_prefix]

            return dec_id, bin_id, psf_meta.member("fParams")  # type: ignore

        raise KeyError("Unknown binning scheme in response file")

    @property
    def declination_bins_lower(self) -> NDArray[np.float64]:
        """Get the simulation declination bin lower edges within ROOT response file"""
        if self.response_ttree_directory.get("DecBins/lowerEdge", None) is not None:
            return (
                self.response_ttree_directory["DecBins/lowerEdge"]
                .array()  # type: ignore
                .to_numpy()
                .astype(np.float64)
            )
        else:
            raise KeyError("DecBins/lowerEdge not found in response file")

    @property
    def declination_bins_upper(self) -> NDArray[np.float64]:
        """Get the simulation declination bin upper edges within ROOT response file"""
        if self.response_ttree_directory.get("DecBins/upperEdge", None) is not None:
            return (
                self.response_ttree_directory["DecBins/upperEdge"]
                .array()  # type: ignore
                .to_numpy()
                .astype(np.float64)
            )
        else:
            raise KeyError("DecBins/upperEdge not found in response file")

    @property
    def declination_bins_center(self) -> NDArray[np.float64]:
        """Get the simulation declination bin centers within ROOT response file"""
        if self.response_ttree_directory.get("DecBins/simdec", None) is not None:
            return (
                self.response_ttree_directory["DecBins/simdec"]
                .array()  # type: ignore
                .to_numpy()
                .astype(np.float64)
            )
        else:
            raise KeyError("DecBins/simdec not found in response file")

    @property
    def analysis_bins(self) -> NDArray[np.string_]:
        """Get the analysis bin names within ROOT response file"""
        if self.response_ttree_directory.get("AnalysisBins/name", None) is not None:
            return (
                self.response_ttree_directory["AnalysisBins/name"]
                .array()  # type: ignore
                .to_numpy()
                .astype(dtype=str)
            )
        if self.response_ttree_directory.get("AnalysisBins/id", None) is not None:
            return (
                self.response_ttree_directory["AnalysisBins/id"]
                .array()  # type: ignore
                .to_numpy()
                .astype(dtype=str)
            )

        raise KeyError("Unknown binning scheme in response file")

    @property
    def log_log_params(self) -> np.ndarray:
        """Read the best-fit params from PSF fit from ROOT file"""
        if self.response_ttree_directory.get("LogLogSpectrum", None) is not None:
            return np.array(
                self.response_ttree_directory["LogLogSpectrum"].member("fParams")  # type: ignore
            )
        raise KeyError("LogLogSpectrum not found in response file")

    @property
    def spectrum_shape(self) -> str:
        """Read the best-fit params from PSF fit from ROOT file"""
        if self.response_ttree_directory.get("LogLogSpectrum", None) is not None:
            return self.response_ttree_directory["LogLogSpectrum"].member("fTitle")  # type: ignore

        raise KeyError("LogLogSpectrum not found in response file")


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

    @staticmethod
    def create_dict_from_results(results):
        """Handles the calculation of the results from the multiprocessing pool"""
        return {(result[0], result[1]): result[2] for result in results}

    @classmethod
    def from_root_file(cls, response_file_name: Path, n_workers: int = 1):
        """Read the response ROOT file. Do not use directly, use the
        hawc_response_factory method instead.

        Parameters
        ----------
        response_file_name : Path
            Response ROOT file path
        n_workers : int
            Number of workers to use for multiprocessing

        Returns
        -------
        HAWCResponse :
            HAWC response file object containing the response bins for declinations
            within the ROOT file

        Raises
        ------
        IOError
            This exception will be raised if the file provided does not exists or is not readable
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
            resp_metadata = ResponseMetaData(response_file_directory)

            # NOTE:Get the Response function basic information
            log_log_params = resp_metadata.log_log_params
            log_log_shape = resp_metadata.spectrum_shape
            analysis_bins_arr = resp_metadata.analysis_bins
            dec_bins_lower_edge = resp_metadata.declination_bins_lower
            dec_bins_upper_edge = resp_metadata.declination_bins_upper
            dec_bins_sim = resp_metadata.declination_bins_center

            dec_bins = list(zip(dec_bins_lower_edge, dec_bins_sim, dec_bins_upper_edge))
            number_of_dec_bins = len(dec_bins_sim)

            args = [
                (response_file_directory, dec_id, bin_id)
                for dec_id in range(number_of_dec_bins)
                for bin_id in analysis_bins_arr
            ]

            with multiprocessing.Pool(processes=n_workers) as executor:
                results = list(executor.starmap(resp_metadata.get_energy_hist, args))
                results_bkg = list(
                    executor.starmap(resp_metadata.get_energy_bkg_hist, args)
                )
                psf_param = list(executor.starmap(resp_metadata.get_psf_params, args))

        energy_hists = cls.create_dict_from_results(results)
        energy_bkgs = cls.create_dict_from_results(results_bkg)
        psf_metas = cls.create_dict_from_results(psf_param)

        # NOTE: Now we have all the info we need to build the response
        # TODO: read only the declinations needed for the ROI
        response_bins = collections.OrderedDict()
        for dec_id, dec_bin in enumerate(dec_bins):
            min_dec, dec_center, max_dec = dec_bin
            response_bins_per_dec = collections.OrderedDict()

            for bin_id in analysis_bins_arr:
                current_hist = energy_hists[(dec_id, bin_id)]
                current_hist_bkg = energy_bkgs[(dec_id, bin_id)]
                current_psf_params = psf_metas[(dec_id, bin_id)]

                this_response_bin = ResponseBin.from_ttree(
                    bin_id,
                    current_hist,
                    current_hist_bkg,
                    psf_fit_params=current_psf_params,
                    log_log_params=log_log_params,
                    log_log_shape=log_log_shape,
                    min_dec=min_dec,
                    dec_center=dec_center,
                    max_dec=max_dec,
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
