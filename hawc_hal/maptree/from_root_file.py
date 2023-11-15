from __future__ import absolute_import

import collections
import multiprocessing
from builtins import str
from pathlib import Path
from typing import Optional, Tuple, Union

import healpy as hp
import numpy as np
import uproot
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from hawc_hal.region_of_interest.healpix_cone_roi import HealpixConeROI
from hawc_hal.region_of_interest.healpix_map_roi import HealpixMapROI

from ..healpix_handling import DenseHealpix, SparseHealpix
from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin

log = setup_logger(__name__)
log.propagate = False


def retrieve_data(args: Tuple[uproot.ReadOnlyDirectory, str]) -> Tuple[str, np.ndarray]:
    """function to iterate over the maptree and read the data and bkg. maps"""
    array_metadata, bin_name = args
    return bin_name, array_metadata.array().to_numpy()


def multiprocessing_data(tree_metadata, bins_labels: np.ndarray, n_workers: int):
    """Carry the multiprocessing of the data and bkg. maps"""
    # with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    #     result = list(executor.map(retrieve_data, tree_metadata, bins_labels))
    with multiprocessing.Pool(processes=n_workers) as executor:
        args = zip(tree_metadata, bins_labels)
        result = list(executor.map(retrieve_data, args))
    return result


def retrieve_results(
    map_infile: uproot.ReadOnlyDirectory,
    bin_branch: str,
    nhit_name_prefix: str,
    n_workers: int,
):
    try:
        data_bins_labels = map_infile[bin_branch].array().to_numpy()
        data_tree_metadata = [
            map_infile[f"{nhit_name_prefix}{name}/data/count"]
            for name in data_bins_labels
        ]
        bkg_tree_metadata = [
            map_infile[f"{nhit_name_prefix}{name}/bkg/count"]
            for name in data_bins_labels
        ]

        result_data = multiprocessing_data(
            data_tree_metadata, data_bins_labels, n_workers
        )
        result_bkg = multiprocessing_data(
            bkg_tree_metadata, data_bins_labels, n_workers
        )
        return result_data, result_bkg

    except Exception as e:
        raise Exception(f"Error reading Maptree: {e}") from e


def from_root_file(
    map_tree_file: Path,
    roi: Union[HealpixConeROI, HealpixMapROI],
    transits: float,
    n_workers: int,
    scheme: int = 0,
):
    """Create a Maptree object from a ROOT file and a ROI.
    Do not use this directly, use map_tree_factory instead.

    Args:
        map_tree_file (str): maptree root file
        roi (HealpixROIBase): region of interest set with HealpixConeROI
        nside (int): HEALPix Nside number
        scheme (int): specify RING or NESTED HEALPix pixel scheme

    Raises:
        IOError: An IOError is raised if the maptree file is corrupted or unable
        to be read
        ValueError: A ValueError is raised if maptree doesn't contain the 'name'
        or 'id' bin naming scheme

    Returns:
        dict: returns a dictionary with names of analysis bins found in Maptree
    """

    # from ..root_handler import open_ROOT_file, root_numpy, tree_to_ndarray

    map_tree_file = sanitize_filename(map_tree_file)

    # Check that they exists and can be read

    if not file_existing_and_readable(map_tree_file):  # pragma: no cover
        raise IOError(f"MapTree {map_tree_file} does not exist or is not readable")

    # Make sure we have a proper ROI (or None)

    assert isinstance(roi, HealpixROIBase) or roi is None, (
        "You have to provide an ROI choosing from the "
        "available ROIs in the region_of_interest module"
    )

    if roi is None:
        log.warning("You have set roi=None, so you are reading the entire sky")

    # Note: Main motivation: root_numpy has been deprecated and it its no longer supported
    # uproot seems like an alternative to this challenge
    # uproot an I/O framework for reading information from ROOT files, so it
    # NOTE: no direct way of reading Nside and HEALPix scheme from ROOT file
    # cannot perform operations on histrograms

    # Read the maptree
    with uproot.open(str(map_tree_file)) as map_infile:
        log.info("Reading Maptree!")

        maptree_durations: np.ndarray = (
            map_infile["BinInfo/totalDuration"].array().to_numpy()
        )

        binning_scheme_name = {
            "BinInfo/name": "nHit",
            "BinInfo/id": "nHit0",
        }

        bin_branch: Optional[str] = None
        nhit_name_prefix: Optional[str] = None
        for bin_info_name, nhit_scheme in binning_scheme_name.items():
            if map_infile.get(bin_info_name, None) is not None:
                nhit_name_prefix = nhit_scheme
                bin_branch = bin_info_name

        data_bins_labels = map_infile[bin_branch].array().to_numpy()

        # Launch processes to speed up the reading of the maptree file
        # NOTE: The number of workers is suggested to be kept equal one less
        # than the number of available cores in the system.
        result_data, result_bkg = retrieve_results(
            map_infile, bin_branch, nhit_name_prefix, n_workers
        )

        # Processes are not guaranteed to preserve order of analysis bin names
        # Organize them into a dictionary for proper readout
        data_dir_array = dict(result_data)
        bkg_dir_array = dict(result_bkg)

    log.info(f"Loaded {map_tree_file}")

    npix_cnt = result_data[0][1].size
    npix_bkg = result_bkg[0][1].size

    # The map-maker underestimate the livetime of bins with low statistic
    # by removing time intervals with zero events. Therefore, the best
    # estimate of the livetime is the maximum of n_transits, which normally
    # happen in the bins with high statistic
    max_duration: float = np.divide(maptree_durations.max(), 24.0)

    # use value of maptree unless otherwise specified by user
    n_transits = max_duration if transits is None else transits
    scale_factor: float = n_transits / max_duration

    nside_cnt: int = hp.pixelfunc.npix2nside(npix_cnt)
    nside_bkg: int = hp.pixelfunc.npix2nside(npix_bkg)

    assert (
        nside_cnt == nside_bkg
    ), "Nside value needs to be the same for counts and bkg. maps"

    assert scheme == 0, "NESTED HEALPix is not currently supported."

    data_analysis_bins = collections.OrderedDict()

    healpix_map_active = np.zeros(hp.nside2npix(nside_cnt))

    if roi is not None:
        active_pixels = roi.active_pixels(
            nside_cnt, system="equatorial", ordering="RING"
        )

        healpix_map_active[active_pixels] = 1.0

        for name in data_bins_labels:
            counts = data_dir_array[name]
            bkg = bkg_dir_array[name]

            counts *= scale_factor
            bkg *= scale_factor

            counts_hpx = SparseHealpix(
                counts[healpix_map_active > 0.0], active_pixels, nside_cnt
            )
            bkg_hpx = SparseHealpix(
                bkg[healpix_map_active > 0.0], active_pixels, nside_bkg
            )

            # Read only elements within the ROI
            this_data_analysis_bin = DataAnalysisBin(
                name,
                counts_hpx,
                bkg_hpx,
                active_pixels_ids=active_pixels,
                n_transits=n_transits,
                scheme="RING",
            )
            data_analysis_bins[name] = this_data_analysis_bin
    else:
        # If no ROI is provided read the entire sky
        for name in data_bins_labels:
            counts = data_dir_array[name]
            bkg = bkg_dir_array[name]

            counts *= scale_factor
            bkg *= scale_factor

            this_data_analysis_bin = DataAnalysisBin(
                name,
                DenseHealpix(counts),
                DenseHealpix(bkg),
                active_pixels_ids=None,
                n_transits=n_transits,
                scheme="RING",
            )

            data_analysis_bins[name] = this_data_analysis_bin

    return data_analysis_bins, n_transits
