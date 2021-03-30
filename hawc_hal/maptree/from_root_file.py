from __future__ import absolute_import
from builtins import map
from builtins import str
from builtins import range
import os
import socket
import collections
import numpy as np

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from threeML.io.logging import setup_logger
log = setup_logger(__name__)
log.propagate = False

from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin

from ..healpix_handling import SparseHealpix, DenseHealpix


def _get_bin_object(f, bin_name, suffix):

    # kludge: some maptrees have bin labels in BinInfo like "0", "1", "2", but then
    # the nHit bin is actually "nHit00, nHit01, nHit02... others instead have
    # labels in BinInfo like "00", "01", "02", and still the nHit bin is nHit00, nHit01
    # thus we need to add a 0 to the former case, but not add it to the latter case

    bin_label = "nHit0%s/%s" % (bin_name, suffix)

    bin_tobject = f.Get(bin_label)

    if not bin_tobject:

        # Try the other way
        bin_label = "nHit%s/%s" % (bin_name, suffix)

        bin_tobject = f.Get(bin_label)

        if not bin_tobject:

            raise IOError("Could not read bin %s" % bin_label)

    return bin_tobject


def from_root_file(map_tree_file, roi, n_transits):
    """
    Create a MapTree object from a ROOT file and a ROI. Do not use this directly, use map_tree_factory instead.

    :param map_tree_file:
    :param roi:
    :param n_transits:
    :return:
    """

    from ..root_handler import open_ROOT_file, root_numpy, tree_to_ndarray

    map_tree_file = sanitize_filename(map_tree_file)

    # Check that they exists and can be read

    if not file_existing_and_readable(map_tree_file):  # pragma: no cover
        raise IOError("MapTree %s does not exist or is not readable" % map_tree_file)

    # Make sure we have a proper ROI (or None)

    assert isinstance(roi, HealpixROIBase) or roi is None, "You have to provide an ROI choosing from the " \
                                                           "available ROIs in the region_of_interest module"

    if roi is None:
        log.warning("You have set roi=None, so you are reading the entire sky")

    # Read map tree
    with open_ROOT_file(str(map_tree_file)) as f:

        # Newer maps use "name" rather than "id"
        try:

            data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "name"))

        except ValueError:

            # Check to see if its an old style maptree 
            try:

                data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "id"))

            except ValueError:
                
                # Give a useful error message
                raise ValueError("Maptree has no Branch: 'id' or 'name' ")

            # If the old style, we need to make them strings
            data_bins_labels = [ str(i) for i in data_bins_labels ]



        # The map-maker underestimate the livetime of bins with low statistic by removing time intervals with
        # zero events. Therefore, the best estimate of the livetime is the maximum of n_transits, which normally
        # happen in the bins with high statistic
        # Alternatively, specify n_transits

        if n_transits is None:
            # A transit is defined as 1 day, and totalDuration is in hours
            # Get the number of transits from bin 0 (as LiFF does)
            map_transits = root_numpy.tree2array(f.Get("BinInfo"), "totalDuration") / 24.0
            use_transits = np.max(map_transits)
        else:
            use_transits = n_transits
        
        n_bins = len(data_bins_labels)

        # These are going to be Healpix maps, one for each data analysis bin_name

        data_analysis_bins = collections.OrderedDict()

        for i in range(n_bins):

            name = data_bins_labels[i]

            data_tobject = _get_bin_object(f, name, "data")

            bkg_tobject = _get_bin_object(f, name, "bkg")

            # Get ordering scheme
            nside = data_tobject.GetUserInfo().FindObject("Nside").GetVal()
            nside_bkg = bkg_tobject.GetUserInfo().FindObject("Nside").GetVal()

            assert nside == nside_bkg

            scheme = data_tobject.GetUserInfo().FindObject("Scheme").GetVal()
            scheme_bkg = bkg_tobject.GetUserInfo().FindObject("Scheme").GetVal()

            assert scheme == scheme_bkg

            assert scheme == 0, "NESTED scheme is not supported yet"

            if roi is not None:

                # Only read the elements in the ROI

                active_pixels = roi.active_pixels(nside, system='equatorial', ordering='RING')

                counts = _read_partial_tree(data_tobject, active_pixels)
                bkg = _read_partial_tree(bkg_tobject, active_pixels)

                counts_hpx = SparseHealpix(counts, active_pixels, nside)
                bkg_hpx = SparseHealpix(bkg, active_pixels, nside)

                this_data_analysis_bin = DataAnalysisBin(name,
                                                         counts_hpx,
                                                         bkg_hpx,
                                                         active_pixels_ids=active_pixels,
                                                         n_transits=use_transits,
                                                         scheme='RING')

            else:

                # Read the entire sky.

                counts = tree_to_ndarray(data_tobject, "count").astype(np.float64)
                bkg = tree_to_ndarray(bkg_tobject, "count").astype(np.float64)

                this_data_analysis_bin = DataAnalysisBin(name,
                                                         DenseHealpix(counts),
                                                         DenseHealpix(bkg),
                                                         active_pixels_ids=None,
                                                         n_transits=use_transits,
                                                         scheme='RING')

            data_analysis_bins[name] = this_data_analysis_bin

    return data_analysis_bins, use_transits


def _read_partial_tree(ttree_instance, elements_to_read):

    # Decide whether to use a smart loading scheme, or just loading the whole thing, based on the
    # number of elements to be read

    from ..root_handler import ROOT, root_numpy, tree_to_ndarray

    if elements_to_read.shape[0] < 500000:

        # Use a smart loading scheme, where we read only the pixels we need

        # The fastest method that I found is to create a TEventList, apply it to the
        # tree, get a copy of the subset and then use ttree2array

        # Create TEventList
        entrylist = ROOT.TEntryList()

        # Add the selections
        _ = list(map(entrylist.Enter, elements_to_read))

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

    return partial_map.astype(np.float64)
