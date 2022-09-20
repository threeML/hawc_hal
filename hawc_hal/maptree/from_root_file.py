from __future__ import absolute_import

import collections
import os
import socket
from builtins import map, range, str
from itertools import count
from pathlib import Path
from token import N_TOKENS

import healpy as hp
import numpy as np
import uproot
from matplotlib.style import library
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

log = setup_logger(__name__)
log.propagate = False

from ..healpix_handling import DenseHealpix, SparseHealpix
from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin


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

        try:

            data_bins_labels = map_infile["BinInfo/name"].array().to_numpy()

        except ValueError:

            try:

                data_bins_labels = map_infile["BinInfo/id"].array().to_numpy()

            except ValueError as exc:

                raise ValueError("Maptree has no Branch: 'id' or 'name'") from exc

        # HACK:workaround method of getting the Nside from maptree
        bin_name = data_bins_labels[0]

        try:
            data_tree_prefix = f"nHit{bin_name}/data/count"
            bkg_tree_prefix = f"nHit{bin_name}/bkg/count"

            npix_cnt = map_infile[data_tree_prefix].array().to_numpy().size
            npix_bkg = map_infile[bkg_tree_prefix].array().to_numpy().size

        except uproot.KeyInFileError:
            data_tree_prefix = f"nHit0{bin_name}/data/count"
            bkg_tree_prefix = f"nHit0{bin_name}/bkg/count"

            npix_cnt = map_infile[data_tree_prefix].array().to_numpy().size
            npix_bkg = map_infile[bkg_tree_prefix].array().to_numpy().size

        # The map-maker underestimate the livetime of bins with low statistic
        # by removing time intervals with zero events. Therefore, the best
        # estimate of the livetime is the maximum of n_transits, which normally
        # happen in the bins with high statistic
        maptree_durations = map_infile["BinInfo/totalDuration"].array()
        n_durations: np.ndarray = np.divide(maptree_durations, 24.0)

        # use value of maptree unless otherwise specified by user
        n_transits = max(n_durations) if transits is None else transits

        # assert n_transits <= max(
        #     n_durations
        # ), "Cannot use a higher value than that of maptree."

        n_bins: int = data_bins_labels.shape[0]
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

        # HACK: simple way of reading the number of active pixels within
        # the define ROI
        if roi is not None:

            active_pixels = roi.active_pixels(
                nside_cnt, system="equatorial", ordering="RING"
            )

            for pix_id in active_pixels:

                healpix_map_active[pix_id] = 1.0

        for i in range(n_bins):

            name = data_bins_labels[i]

            try:

                data_tree_prefix = f"nHit{name}/data/count"
                bkg_tree_prefix = f"nHit{name}/bkg/count"

                # if using number of transitions different from the maptree
                # then make sure counts are scaled accordingly.
                counts: np.ndarray = map_infile[data_tree_prefix].array().to_numpy() * (
                    n_transits / max(n_durations)
                )
                bkg: np.ndarray = map_infile[bkg_tree_prefix].array().to_numpy() * (
                    n_transits / max(n_durations)
                )

            except uproot.KeyInFileError:

                # Sometimes, names of bins carry an extra zero
                data_tree_prefix = f"nHit0{name}/data/count"
                bkg_tree_prefix = f"nHit0{name}/bkg/count"

                counts = map_infile[data_tree_prefix].array().to_numpy()
                bkg = map_infile[bkg_tree_prefix].array().to_numpy()

            # Read only elements within the ROI
            if roi is not None:

                # HACK: first attempt at reading only a partial map specified
                # by the active pixel ids.

                counts_hpx = SparseHealpix(
                    counts[healpix_map_active > 0], active_pixels, nside_cnt
                )
                bkg_hpx = SparseHealpix(
                    bkg[healpix_map_active > 0], active_pixels, nside_cnt
                )

                this_data_analysis_bin = DataAnalysisBin(
                    name,
                    counts_hpx,
                    bkg_hpx,
                    active_pixels_ids=active_pixels,
                    n_transits=n_transits,
                    scheme="RING",
                )

            # Read the whole sky
            else:

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
