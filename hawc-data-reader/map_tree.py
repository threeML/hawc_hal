import numpy as np
import healpy as hp
import root_numpy
import pandas as pd
import six

import ROOT

from threeML.io.cern_root_utils.io_utils import open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray
from threeML.io.rich_display import display
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from region_of_interest import HealpixROIBase
from sparse_healpix import SparseHealpix, DenseHealpix
from special_values import UNSEEN

import astropy.units as u


class DataAnalysisBin(object):

    def __init__(self, name, observation_hpx_map, background_hpx_map, active_pixels_ids, scheme='RING'):

        # Get nside

        self._nside = observation_hpx_map.nside

        nside_bkg = background_hpx_map.nside

        assert self._nside == nside_bkg, "Observation and background maps have " \
                                         "different nside (%i vs %i)" % (self._nside, nside_bkg)

        self._npix = observation_hpx_map.npix

        # Store healpix maps
        self._observation_hpx_map = observation_hpx_map

        self._background_hpx_map = background_hpx_map

        # Store the active pixels (i.e., the pixels that are within the selected ROI)
        self._active_pixels_ids = active_pixels_ids

        # Store name and scheme
        self._name = name

        assert scheme in ['RING', 'NEST'], "Scheme must be either RING or NEST"

        self._scheme = scheme

    @property
    def name(self):

        return self._name

    @property
    def scheme(self):

        return self._scheme

    @property
    def nside(self):

        return self._nside

    @property
    def npix(self):
        return self._npix

    @property
    def observation_map(self):

        return self._observation_hpx_map

    @property
    def background_map(self):

        return self._background_hpx_map

    @property
    def active_pixels_ids(self):

        return self._active_pixels_ids



class MapTree(object):

    def __init__(self, map_tree_file, roi):

        # Sanitize files in input (expand variables and so on)

        map_tree_file = sanitize_filename(map_tree_file)

        # Check that they exists and can be read

        if not file_existing_and_readable(map_tree_file):

            raise IOError("MapTree %s does not exist or is not readable" % map_tree_file)

        # Make sure we have a proper ROI (or None)

        assert isinstance(roi, HealpixROIBase) or roi is None, "You have to provide an ROI choosing from the " \
                                                               "available ROIs in the region_of_interest module"

        if roi is None:

            custom_warnings.warn("You have set roi=None, so you are reading the entire sky")

        # This dictionary will contain the selected pixels as function of nside,
        # for faster access (see the _get_entry_list method)
        self._entry_list_cache = {}

        # Read map tree

        with open_ROOT_file(map_tree_file) as f:

            self._data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "name"))

            # A transit is defined as 1 day, and totalDuration is in hours

            self._n_transits = np.average(root_numpy.tree2array(f.Get("BinInfo"), "totalDuration")) / 24.0

            n_bins = len(self._data_bins_labels)

            # These are going to be Healpix maps, one for each data analysis bin

            self._data_analysis_bins = []

            for i in range(n_bins):

                name = self._data_bins_labels[i]

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

                    elements_to_read = roi.active_pixels(nside, system='equatorial', ordering='RING')

                    counts = self._read_partial_tree(nside, f.Get(bin_label), elements_to_read)
                    bkg = self._read_partial_tree(nside, f.Get(bkg_label), elements_to_read)

                    this_data_analysis_bin = DataAnalysisBin(name,
                                                             SparseHealpix(counts, copy=True),
                                                             SparseHealpix(bkg, copy=True),
                                                             active_pixels_ids=elements_to_read,
                                                             scheme='RING')

                else:

                    # Read the entire sky.

                    counts = tree_to_ndarray(f.Get(bin_label), "count").astype(np.float32)
                    bkg = tree_to_ndarray(f.Get(bkg_label), "count").astype(np.float32)

                    this_data_analysis_bin = DataAnalysisBin(name,
                                                             DenseHealpix(counts),
                                                             DenseHealpix(bkg),
                                                             active_pixels_ids=None,
                                                             scheme='RING')

                self._data_analysis_bins.append(this_data_analysis_bin)

    @property
    def n_transits(self):

        return self._n_transits

    def __iter__(self):
        """
        This allows to loop over the analysis bins as in:

        for analysis_bin in maptree:

            ... do something ...

        :return: analysis bin iterator
        """

        for analysis_bin in self._data_analysis_bins:

            yield analysis_bin

    def __getitem__(self, item):
        """
        This allows to access the analysis bins as:

        first_analysis_bin = maptree[0]

        or by name:

        first_analysis_bin = maptree["bin 0"]

        :param item: integer for serial access, string for access by name
        :return: the analysis bin
        """

        if isinstance(item, six.string_types):

            try:

                id = self._data_bins_labels.index(item)

            except ValueError:

                raise KeyError("There is no analysis bin named %s" % item)

            else:

                return self._data_analysis_bins[id]

        else:

            try:

                return self._data_analysis_bins[item]

            except IndexError:

                raise IndexError("Analysis bin with index %i does not exist" % (item))

    def __len__(self):

        return len(self._data_analysis_bins)

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

        # Decide whether to use a smart loading scheme, or just loading the whole thing, based on the
        # number of elements to be read

        if elements_to_read.shape[0] < 500000:

            # Use a smart loading scheme, where we read only the pixels we need

            # First create a dense array of the appropriate size (note that the value which means "unobserved"
            # is UNSEEN, which fits in a np.float32)
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

            # Now overwrite the UNSEEN with the true value of the pixels
            # we just read. All the others will stay UNSEEN

            for i, idx in enumerate(elements_to_read):

                dense[idx] = elements[i]

        else:

            # The smart scheme starts to become slower than the brute force approach, so let's read the whole thing
            dense = tree_to_ndarray(ttree_instance, "count").astype(np.float32)

            # Now let's put the pixels we shouldn't have read to UNSEEN
            mask = np.ones(dense.shape[0], dtype=bool)
            mask[elements_to_read] = False

            dense[mask] = UNSEEN

        return dense

    def display(self):

        df = pd.DataFrame()

        df['Bin'] = self._data_bins_labels
        df['Nside'] = map(lambda x:x.nside, self._data_analysis_bins)
        df['Scheme'] = map(lambda x:x.scheme, self._data_analysis_bins)

        # Compute observed counts, background counts, how many pixels we have in the ROI and
        # the sky area they cover
        n_bins = len(self._data_bins_labels)

        obs_counts = np.zeros(n_bins)
        bkg_counts = np.zeros_like(obs_counts)
        n_pixels = np.zeros(n_bins, dtype=int)
        sky_area = np.zeros_like(obs_counts)

        size = 0

        for i, analysis_bin in enumerate(self._data_analysis_bins):

            sparse_obs = analysis_bin.observation_map.as_sparse()
            sparse_bkg = analysis_bin.background_map.as_sparse()

            size += sparse_obs.nbytes
            size += sparse_bkg.nbytes

            obs_counts[i] = sparse_obs.sum()
            bkg_counts[i] = sparse_bkg.sum()
            n_pixels[i] = sparse_obs.shape[0]
            sky_area[i] = n_pixels[i] * analysis_bin.observation_map.pixel_area

        df['Obs counts'] = obs_counts
        df['Bkg counts'] = bkg_counts
        df['obs/bkg'] = obs_counts / bkg_counts
        df['Pixels in ROI'] = n_pixels
        df['Area (deg^2)'] = sky_area

        display(df)

        print("This Map Tree contains %.3f transits" % self.n_transits)
        print("Total data size: %s" % (size * u.byte).to(u.megabyte))