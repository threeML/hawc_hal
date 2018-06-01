import numpy as np
import os
import socket

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.exceptions.custom_exceptions import custom_warnings

from ..region_of_interest import HealpixROIBase
from data_analysis_bin import DataAnalysisBin

from ..healpix_handling import SparseHealpix, DenseHealpix

try:

    import ROOT
    from threeML.io.cern_root_utils.io_utils import open_ROOT_file
    from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray

    ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )

    import root_numpy

except ImportError:  # pragma: no cover

    pass


def from_root_file(map_tree_file, roi):
    """
    Create a MapTree object from a ROOT file and a ROI. Do not use this directly, use map_tree_factory instead.

    :param map_tree_file:
    :param roi:
    :return:
    """

    map_tree_file = sanitize_filename(map_tree_file)

    # Check that they exists and can be read

    if not file_existing_and_readable(map_tree_file):  # pragma: no cover
        raise IOError("MapTree %s does not exist or is not readable" % map_tree_file)

    # Make sure we have a proper ROI (or None)

    assert isinstance(roi, HealpixROIBase) or roi is None, "You have to provide an ROI choosing from the " \
                                                           "available ROIs in the region_of_interest module"

    if roi is None:
        custom_warnings.warn("You have set roi=None, so you are reading the entire sky")

    # Read map tree

    with open_ROOT_file(map_tree_file) as f:

        data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "name"))

        # A transit is defined as 1 day, and totalDuration is in hours
        # Get the number of transit from bin 0 (as LiFF does)

        n_transits = root_numpy.tree2array(f.Get("BinInfo"), "totalDuration") / 24.0

        n_bins = len(data_bins_labels)

        # These are going to be Healpix maps, one for each data analysis bin_name

        data_analysis_bins = []

        for i in range(n_bins):

            name = data_bins_labels[i]

            bin_label = "nHit0%s/%s" % (name, "data")

            bkg_label = "nHit0%s/%s" % (name, "bkg")

            # Get ordering scheme
            nside = f.Get(bin_label).GetUserInfo().FindObject("Nside").GetVal()
            nside_bkg = f.Get(bkg_label).GetUserInfo().FindObject("Nside").GetVal()

            assert nside == nside_bkg

            scheme = f.Get(bin_label).GetUserInfo().FindObject("Scheme").GetVal()
            scheme_bkg = f.Get(bkg_label).GetUserInfo().FindObject("Scheme").GetVal()

            assert scheme == scheme_bkg

            assert scheme == 0, "NESTED scheme is not supported yet"

            if roi is not None:

                # Only read the elements in the ROI

                active_pixels = roi.active_pixels(nside, system='equatorial', ordering='RING')

                counts = _read_partial_tree(f.Get(bin_label), active_pixels)
                bkg = _read_partial_tree(f.Get(bkg_label), active_pixels)

                this_data_analysis_bin = DataAnalysisBin(name,
                                                         SparseHealpix(counts, active_pixels, nside),
                                                         SparseHealpix(bkg, active_pixels, nside),
                                                         active_pixels_ids=active_pixels,
                                                         n_transits=n_transits[i],
                                                         scheme='RING')

            else:

                # Read the entire sky.

                counts = tree_to_ndarray(f.Get(bin_label), "count").astype(np.float64)
                bkg = tree_to_ndarray(f.Get(bkg_label), "count").astype(np.float64)

                this_data_analysis_bin = DataAnalysisBin(name,
                                                         DenseHealpix(counts),
                                                         DenseHealpix(bkg),
                                                         active_pixels_ids=None,
                                                         n_transits=n_transits[i],
                                                         scheme='RING')

            data_analysis_bins.append(this_data_analysis_bin)

    return data_bins_labels, data_analysis_bins


def _read_partial_tree(ttree_instance, elements_to_read):

    # Decide whether to use a smart loading scheme, or just loading the whole thing, based on the
    # number of elements to be read

    if elements_to_read.shape[0] < 500000:

        # Use a smart loading scheme, where we read only the pixels we need

        # The fastest method that I found is to create a TEventList, apply it to the
        # tree, get a copy of the subset and then use ttree2array

        # Create TEventList
        entrylist = ROOT.TEntryList()

        # Add the selections
        _ = map(entrylist.Enter, elements_to_read)

        # Apply the EntryList to the tree
        ttree_instance.SetEntryList(entrylist)

        # Get copy of the subset
        # We need to create a dumb TFile to silence a lot of warnings from ROOT
        # Get a filename for this process
        dumb_tfile_name = "__dumb_tfile_%s_%s.root" % (os.getpid(), socket.gethostname())
        dumb_tfile = ROOT.TFile(dumb_tfile_name, "RECREATE")
        new_tree = ttree_instance.CopyTree("")

        # Actually read it from disk
        partial_map = root_numpy.tree2array(new_tree, "count").astype(np.float64)

        # Now reset the entry list
        ttree_instance.SetEntryList(0)

        dumb_tfile.Close()
        os.remove(dumb_tfile_name)

    else:

        # The smart scheme starts to become slower than the brute force approach, so let's read the whole thing
        partial_map = tree_to_ndarray(ttree_instance, "count").astype(np.float64)

        assert partial_map.shape[0] >= elements_to_read.shape[0], "Trying to read more pixels than present in TTree"

        # Unless we have read the whole sky, let's remove the pixels we shouldn't have read

        if elements_to_read.shape[0] != partial_map.shape[0]:

            # Now let's remove the pixels we shouldn't have read
            partial_map = partial_map[elements_to_read]

    return partial_map