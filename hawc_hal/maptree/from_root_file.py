from __future__ import absolute_import
from builtins import map
from builtins import str
from builtins import range
from itertools import count
import os
from pathlib import Path
import socket
import collections
from token import N_TOKENS
import healpy as hp
from matplotlib.style import library
import numpy as np
import uproot

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from threeML.io.logging import setup_logger

log = setup_logger(__name__)
log.propagate = False

from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin

from ..healpix_handling import SparseHealpix, DenseHealpix

# ! As of right now, any code with ROOT functionality is commented,
# ! but should be removed in the future.
# def _get_bin_object(f, bin_name, suffix):

#     # kludge: some maptrees have bin labels in BinInfo like "0", "1", "2", but then
#     # the nHit bin is actually "nHit00, nHit01, nHit02... others instead have
#     # labels in BinInfo like "00", "01", "02", and still the nHit bin is nHit00, nHit01
#     # thus we need to add a 0 to the former case, but not add it to the latter case

#     # bin_label = "nHit0%s/%s" % (bin_name, suffix)
#     bin_label = f"nHit0{bin_name}/{suffix}"

#     bin_tobject = f.Get(bin_label)

#     if not bin_tobject:

#         # Try the other way
#         # bin_label = "nHit%s/%s" % (bin_name, suffix)
#         bin_label = f"nHit{bin_name}/{suffix}"

#         bin_tobject = f.Get(bin_label)

#         if not bin_tobject:

#             # raise IOError("Could not read bin %s" % bin_label)
#             raise IOError(f"Could not read bin {bin_label}")

#     return bin_tobject


def from_root_file(
    map_tree_file: Path,
    roi: HealpixROIBase,
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
        # raise IOError("MapTree %s does not exist or is not readable" % map_tree_file)
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

            # data_bins_labels = map_infile["BinInfo"]["name"].array().to_numpy()
            data_bins_labels = map_infile["BinInfo/name"].array().to_numpy()

        except ValueError:

            try:

                # data_bins_labels = map_infile["BinInfo"]["id"].array().to_numpy()
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
            # npix_cnt = (
            # map_infile[f"nHit{bin_name}"]["data"]["count"].array().to_numpy().size
            # )
            # npix_bkg = (
            # map_infile[f"nHit{bin_name}"]["bkg"]["count"].array().to_numpy().size
            # )

        except uproot.KeyInFileError:
            data_tree_prefix = f"nHit0{bin_name}/data/count"
            bkg_tree_prefix = f"nHit0{bin_name}/bkg/count"

            npix_cnt = map_infile[data_tree_prefix].array().to_numpy().size
            npix_bkg = map_infile[bkg_tree_prefix].array().to_numpy().size

            # npix_cnt = (
            # map_infile[f"nHit0{bin_name}"]["data"]["count"].array().to_numpy().size
            # )
            # npix_bkg = (
            # map_infile[f"nHit0{bin_name}"]["bkg"]["count"].array().to_numpy().size
            # )

        # The map-maker underestimate the livetime of bins with low statistic
        # by removing time intervals with zero events. Therefore, the best
        # estimate of the livetime is the maximum of n_transits, which normally
        # happen in the bins with high statistic
        # n_durations = np.divide(map_infile["BinInfo"]["totalDuration"].array(), 24.0)
        maptree_durations = map_infile["BinInfo/totalDuration"].array()
        n_durations = np.divide(maptree_durations, 24.0)
        n_transits: float = max(n_durations)

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

                counts = map_infile[data_tree_prefix].array().to_numpy()
                bkg = map_infile[bkg_tree_prefix].array().to_numpy()
                # counts = map_infile[f"nHit{name}"]["data"]["count"].array().to_numpy()
                # bkg = map_infile[f"nHit{name}"]["bkg"]["count"].array().to_numpy()

            except uproot.KeyInFileError:

                # Sometimes, names of bins carry an extra zero
                data_tree_prefix = f"nHit0{name}/data/count"
                bkg_tree_prefix = f"nHit0{name}/bkg/count"

                counts = map_infile[data_tree_prefix].array().to_numpy()
                bkg = map_infile[bkg_tree_prefix].array().to_numpy()
                # counts = map_infile[f"nHit0{name}"]["data"]["count"].array().to_numpy()
                # bkg = map_infile[f"nHit0{name}"]["bkg"]["count"].array().to_numpy()

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

                # try:

                #     counts = (
                #         map_infile[f"nHit{name}"]["data"]["count"].array().to_numpy()
                #     )
                #     bkg = map_infile[f"nHit{name}"]["bkg"]["count"].array().to_numpy()

                # except uproot.KeyInFileError:

                #     # Sometimes, names of bins carry an extra zero
                #     counts = (
                #         map_infile[f"nHit0{name}"]["data"]["count"].array().to_numpy()
                #     )
                #     bkg = map_infile[f"nHit0{name}"]["bkg"]["count"].array().to_numpy()

                this_data_analysis_bin = DataAnalysisBin(
                    name,
                    DenseHealpix(counts),
                    DenseHealpix(bkg),
                    active_pixels_ids=None,
                    n_transits=n_transits,
                    scheme="RING",
                )

            data_analysis_bins[name] = this_data_analysis_bin

            # # Read map tree
            # with open_ROOT_file(str(map_tree_file)) as f:

            #     # Newer maps use "name" rather than "id"
            #     try:

            #         data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "name"))

            #     except ValueError:

            #         # Check to see if its an old style maptree
            #         try:

            #             data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "id"))

            #         except ValueError:

            #             # Give a useful error message
            #             raise ValueError("Maptree has no Branch: 'id' or 'name' ")

            #         # If the old style, we need to make them strings
            #         data_bins_labels = [str(i) for i in data_bins_labels]

            #     # A transit is defined as 1 day, and totalDuration is in hours
            #     # Get the number of transit from bin 0 (as LiFF does)

            #     n_transits = root_numpy.tree2array(f.Get("BinInfo"), "totalDuration") / 24.0

            #     # The map-maker underestimate the livetime of bins with low statistic by removing time intervals with
            #     # zero events. Therefore, the best estimate of the livetime is the maximum of n_transits, which normally
            #     # happen in the bins with high statistic
            #     n_transits = max(n_transits)

            #     n_bins = len(data_bins_labels)

            #     # These are going to be Healpix maps, one for each data analysis bin_name

            #     data_analysis_bins = collections.OrderedDict()

            #     for i in range(n_bins):

            #         name = data_bins_labels[i]

            #         data_tobject = _get_bin_object(f, name, "data")

            #         bkg_tobject = _get_bin_object(f, name, "bkg")

            #         # Get ordering scheme
            #         nside = data_tobject.GetUserInfo().FindObject("Nside").GetVal()
            #         nside_bkg = bkg_tobject.GetUserInfo().FindObject("Nside").GetVal()

            #         assert nside == nside_bkg

            #         scheme = data_tobject.GetUserInfo().FindObject("Scheme").GetVal()
            #         scheme_bkg = bkg_tobject.GetUserInfo().FindObject("Scheme").GetVal()

            #         assert scheme == scheme_bkg

            #         assert scheme == 0, "NESTED scheme is not supported yet"

            #         if roi is not None:

            #             # Only read the elements in the ROI

            #             active_pixels = roi.active_pixels(
            #                 nside, system="equatorial", ordering="RING"
            #             )

            #             counts = _read_partial_tree(data_tobject, active_pixels)
            #             bkg = _read_partial_tree(bkg_tobject, active_pixels)

            #             counts_hpx = SparseHealpix(counts, active_pixels, nside)
            #             bkg_hpx = SparseHealpix(bkg, active_pixels, nside)

            #             this_data_analysis_bin = DataAnalysisBin(
            #                 name,
            #                 counts_hpx,
            #                 bkg_hpx,
            #                 active_pixels_ids=active_pixels,
            #                 n_transits=n_transits,
            #                 scheme="RING",
            #             )

            #         else:

            #             # Read the entire sky.

            #             counts = tree_to_ndarray(data_tobject, "count").astype(np.float64)
            #             bkg = tree_to_ndarray(bkg_tobject, "count").astype(np.float64)

            #             this_data_analysis_bin = DataAnalysisBin(
            #                 name,
            #                 DenseHealpix(counts),
            #                 DenseHealpix(bkg),
            #                 active_pixels_ids=None,
            #                 n_transits=n_transits,
            #                 scheme="RING",
            #             )
            #
            # data_analysis_bins[name] = this_data_analysis_bin
    return data_analysis_bins


# def _read_partial_tree(ttree_instance, elements_to_read):

#     # Decide whether to use a smart loading scheme, or just loading the whole thing, based on the
#     # number of elements to be read

#     from ..root_handler import ROOT, root_numpy, tree_to_ndarray

#     if elements_to_read.shape[0] < 500000:

#         # Use a smart loading scheme, where we read only the pixels we need

#         # The fastest method that I found is to create a TEventList, apply it to the
#         # tree, get a copy of the subset and then use ttree2array

#         # Create TEventList
#         entrylist = ROOT.TEntryList()

#         # Add the selections
#         _ = list(map(entrylist.Enter, elements_to_read))

#         # Apply the EntryList to the tree
#         ttree_instance.SetEntryList(entrylist)

#         # Get copy of the subset
#         # We need to create a dumb TFile to silence a lot of warnings from ROOT
#         # Get a filename for this process
#         dumb_tfile_name = "__dumb_tfile_%s_%s.root" % (
#             os.getpid(),
#             socket.gethostname(),
#         )
#         dumb_tfile = ROOT.TFile(dumb_tfile_name, "RECREATE")
#         new_tree = ttree_instance.CopyTree("")

#         # Actually read it from disk
#         partial_map = root_numpy.tree2array(new_tree, "count").astype(np.float64)

#         # Now reset the entry list
#         ttree_instance.SetEntryList(0)

#         dumb_tfile.Close()
#         os.remove(dumb_tfile_name)

#     else:

#         # The smart scheme starts to become slower than the brute force approach, so let's read the whole thing
#         partial_map = tree_to_ndarray(ttree_instance, "count").astype(np.float64)

#         assert (
#             partial_map.shape[0] >= elements_to_read.shape[0]
#         ), "Trying to read more pixels than present in TTree"

#         # Unless we have read the whole sky, let's remove the pixels we shouldn't have read

#         if elements_to_read.shape[0] != partial_map.shape[0]:

#             # Now let's remove the pixels we shouldn't have read
#             partial_map = partial_map[elements_to_read]

#     return partial_map.astype(np.float64)
