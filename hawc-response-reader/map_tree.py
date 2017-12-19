import numpy as np
import healpy as hp
import root_numpy
import pandas as pd
import ROOT

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tgraph_to_arrays, th2_to_arrays, tree_to_ndarray
from threeML.io.rich_display import display
from threeML.exceptions.custom_exceptions import custom_warnings

from region_of_interest import HealpixROIBase

import astropy.units as u


UNSEEN = hp.UNSEEN


class MapTree(object):

    def __init__(self, map_tree_file, roi):

        assert isinstance(roi, HealpixROIBase) or roi is None, "You have to provide an ROI from the available ROIs in the " \
                                                               "region_of_interest module"

        if roi is None:

            custom_warnings.warn("You have set roi=None, so you are reading the entire sky")

        with open_ROOT_file(map_tree_file) as f:

            self._data_bins_labels = root_numpy.tree2array(f.Get("BinInfo"), "name")

            n_bins = len(self._data_bins_labels)

            # These are going to be Healpix maps, one for each data analysis bin

            self._data_bins_obs = []
            self._data_bins_bkg = []
            self._bin_info = []

            self._entry_list_cache = {}

            for i in range(n_bins):

                bin_label = "nHit0%s/%s" % (self._data_bins_labels[i], "data")

                bkg_label = "nHit0%s/%s" % (self._data_bins_labels[i], "bkg")

                # Get NSIDE and ordering scheme
                nside = f.Get(bin_label).GetUserInfo().FindObject("Nside").GetVal()
                nside_bkg = f.Get(bkg_label).GetUserInfo().FindObject("Nside").GetVal()

                assert nside == nside_bkg

                scheme = f.Get(bin_label).GetUserInfo().FindObject("Scheme").GetVal()
                scheme_bkg = f.Get(bkg_label).GetUserInfo().FindObject("Scheme").GetVal()

                assert scheme == scheme_bkg

                assert scheme == 0, "NESTED scheme is not supported yet"

                if roi is not None:

                    # Only read the elements in the ROI

                    elements_to_read = roi.active_pixels(nside, system='equatorial', ordering='RING')

                    counts = self._read_partial_tree(nside, f.Get(bin_label), elements_to_read)
                    bkg = self._read_partial_tree(nside, f.Get(bkg_label), elements_to_read)

                    sparse = True

                else:

                    # Read the entire sky.

                    counts = tree_to_ndarray(f.Get(bin_label), "count").astype(np.float32)
                    bkg = tree_to_ndarray(f.Get(bkg_label), "count").astype(np.float32)

                    sparse = False

                self._bin_info.append((nside, scheme, sparse))
                self._data_bins_obs.append(counts)
                self._data_bins_bkg.append(bkg)

    def _get_entry_list(self, nside, elements_to_read):

        # Here we assume that the ROI does not change as a function of analysis bin
        # Therefore, once nside is the same, the selection must be the same and we
        # can cache it

        if nside in self._entry_list_cache:

            return self._entry_list_cache[nside]

        else:

            # Create TEventList
            entrylist = ROOT.TEntryList()

            # Add the selections
            _ = map(entrylist.Enter, elements_to_read)

            self._entry_list_cache[nside] = entrylist

            return entrylist

    def _read_partial_tree(self, nside, ttree_instance, elements_to_read):

        if elements_to_read.shape[0] < 500000:

            # Use a smart loading scheme, where we read only the pixels we need

            # First create a dense array of the appropriate size (note that the value which means "unobserved" in healpy
            # is hp.UNSEEN, which fits in a np.float32)
            dense = np.zeros(hp.nside2npix(nside), dtype=np.float32) + UNSEEN

            # Now read in only the requested elements

            # The fastest method that I found is to create a TEventList, apply it to the
            # tree, get a copy of the subset and then use ttree2array

            entrylist = self._get_entry_list(nside, elements_to_read)

            # Apply the EntryList to the tree
            ttree_instance.SetEntryList(entrylist)

            # Get copy of the subset
            new_tree = ttree_instance.CopyTree("")

            # Actually read it from disk
            elements = root_numpy.tree2array(new_tree, "count")

            for i, idx in enumerate(elements_to_read):

                dense[idx] = elements[i]

        else:

            # The smart scheme starts to become slower than the brute force approach, so let's read the whole thing
            dense = tree_to_ndarray(ttree_instance, "count").astype(np.float32)

            # Now let's put the pixels we shouldn't have read to UNSEEN
            mask = np.ones(dense.shape[0], dtype=bool)
            mask[elements_to_read] = False

            dense[mask] = UNSEEN

        # Convert to a sparse representation to save memory

        sparse = pd.SparseArray(dense, fill_value=UNSEEN, kind='block', copy=True)

        return sparse

    def display(self):

        df = pd.DataFrame()

        df['Bin'] = self._data_bins_labels
        df['Nside'] = map(lambda x:x[0], self._bin_info)
        df['Scheme'] = map(lambda x:x[1], self._bin_info)

        # Compute observed counts, background counts, how many pixels we have in the ROI and
        # the sky area they cover
        n_bins = len(self._data_bins_labels)

        obs_counts = np.zeros(n_bins)
        bkg_counts = np.zeros_like(obs_counts)
        n_pixels = np.zeros(n_bins, dtype=int)
        sky_area = np.zeros_like(obs_counts)

        for bin_id in range(n_bins):

            this_obs = self._data_bins_obs[bin_id]
            this_bkg = self._data_bins_bkg[bin_id]

            nside = self._bin_info[bin_id][0]
            is_sparse = self._bin_info[bin_id][2]

            if is_sparse:

                # We need to use to_dense otherwise the mask will be all True
                # However, since we have already verified during the reading that the nside
                # of the bkg and of the observed map are the same, we do not need to test
                # also on the background, but we can re-use the indexes from the obs

                selected_pixels_idx = (this_obs.to_dense() != UNSEEN)  # type: np.ndarray

                obs_counts[bin_id] = np.sum(this_obs[selected_pixels_idx])
                bkg_counts[bin_id] = np.sum(this_bkg[selected_pixels_idx])

                n_pixels[bin_id] = selected_pixels_idx.sum()

            else:

                # We have read the entire sky, no need to make subselections

                obs_counts[bin_id] = this_obs.sum()
                bkg_counts[bin_id] = this_bkg.sum()

                n_pixels[bin_id] = hp.nside2npix(nside)

            sky_area[bin_id] = hp.nside2pixarea(nside, degrees=True) * n_pixels[bin_id]

        df['Obs counts'] = obs_counts
        df['Bkg counts'] = bkg_counts
        df['obs/bkg'] = obs_counts / bkg_counts
        df['Pixels in ROI'] = n_pixels
        df['Area (deg^2)'] = sky_area

        display(df)

        size = 0

        for data, bkg in zip(self._data_bins_obs, self._data_bins_bkg):

            size += data.nbytes
            size += bkg.nbytes

        print("Total data size: %s" % (size * u.byte).to(u.megabyte))