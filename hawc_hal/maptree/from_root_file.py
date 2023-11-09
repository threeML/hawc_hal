from __future__ import absolute_import

import collections
import concurrent.futures
import itertools
from builtins import str
from pathlib import Path

import healpy as hp
import numpy as np
import uproot
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from ..healpix_handling import DenseHealpix, SparseHealpix
from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin

log = setup_logger(__name__)
log.propagate = False


def get_array(tree: uproot.ReadOnlyDirectory, prefix: str) -> np.ndarray:
    """function to iterate over the maptree and read the data and bkg. maps"""
    return tree[prefix].array().to_numpy()


def from_root_file(
    map_tree_file: Path,
    roi: HealpixROIBase,
    transits: float,
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

        new_bin_info_name: str = "BinInfo/name"
        legacy_bin_info_name: str = "BinInfo/id"

        if map_infile.get(new_bin_info_name, None) is not None:
            bin_names = map_infile[new_bin_info_name]
            nhit_naming_scheme: str = "nHit"

        else:
            bin_names = map_infile[legacy_bin_info_name]

            nhit_naming_scheme: str = "nHit0"

        data_bins_labels = bin_names.array().to_numpy()

        data_tree_prefixes = [
            f"{nhit_naming_scheme}{name}/data/count" for name in data_bins_labels
        ]
        bkg_tree_prefixes = [
            f"{nhit_naming_scheme}{name}/bkg/count" for name in data_bins_labels
        ]

        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            data_arrays = list(
                executor.map(
                    get_array, itertools.repeat(map_infile), data_tree_prefixes
                )
            )
            bkg_arrays = list(
                executor.map(get_array, itertools.repeat(map_infile), bkg_tree_prefixes)
            )
    log.info(f"Loaded {map_tree_file}")

    npix_cnt = data_arrays[0].size
    npix_bkg = bkg_arrays[0].size

    # The map-maker underestimate the livetime of bins with low statistic
    # by removing time intervals with zero events. Therefore, the best
    # estimate of the livetime is the maximum of n_transits, which normally
    # happen in the bins with high statistic
    max_duration: float = np.divide(maptree_durations.max(), 24.0)

    # use value of maptree unless otherwise specified by user
    n_transits = max_duration if transits is None else transits
    scale_factor: float = n_transits / max_duration

    # data_bins_labels.shape[0]
    nside_cnt: int = hp.pixelfunc.npix2nside(npix_cnt)
    nside_bkg: int = hp.pixelfunc.npix2nside(npix_bkg)

    # so far, a value of Nside of 1024  (perhaps will change) and a
    # RING HEALPix
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

        for name, counts, bkg in zip(data_bins_labels, data_arrays, bkg_arrays):
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
        for name, counts, bkg in zip(data_bins_labels, data_arrays, bkg_arrays):
            # Read the whole sky
            this_data_analysis_bin = DataAnalysisBin(
                name,
                DenseHealpix(counts),
                DenseHealpix(bkg),
                active_pixels_ids=None,
                n_transits=n_transits,
                scheme="RING",
            )

            data_analysis_bins[name] = this_data_analysis_bin
    log.info("Finished iterating over maptree")

    return data_analysis_bins, n_transits
