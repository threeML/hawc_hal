import numpy as np
import root_numpy
import pandas as pd
import six
import os

import ROOT
ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )


from threeML.io.cern_root_utils.io_utils import open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray
from threeML.io.rich_display import display
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from region_of_interest import HealpixROIBase
from hawc_hal.healpix_handling.sparse_healpix import SparseHealpix, DenseHealpix

import astropy.units as u


class DataAnalysisBin(object):

    def __init__(self, name, observation_hpx_map, background_hpx_map, active_pixels_ids, scheme='RING'):

        # Get nside

        self._nside = observation_hpx_map.nside

        nside_bkg = background_hpx_map.nside

        assert self._nside == nside_bkg, "Observation and background maps have " \
                                         "different nside (%i vs %i)" % (self._nside, nside_bkg)

        self._npix = observation_hpx_map.npix

        # Store healpix maps, clipping them to a small number so no pixel will be zero
        self._observation_hpx_map = np.clip(observation_hpx_map, 1e-52, None)

        self._background_hpx_map = np.clip(background_hpx_map, 1e-52, None)

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


def map_tree_factory(map_tree_file, roi):

    return MapTree(map_tree_file, roi)


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

        # Read map tree

        with open_ROOT_file(map_tree_file) as f:

            self._data_bins_labels = list(root_numpy.tree2array(f.Get("BinInfo"), "name"))

            # A transit is defined as 1 day, and totalDuration is in hours
            # Get the number of transit from bin 0 (as LiFF does)

            self._n_transits = root_numpy.tree2array(f.Get("BinInfo"), "totalDuration")[0] / 24.0

            n_bins = len(self._data_bins_labels)

            # These are going to be Healpix maps, one for each data analysis bin_name

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

                    active_pixels = roi.active_pixels(nside, system='equatorial', ordering='RING')

                    counts = self._read_partial_tree(nside, f.Get(bin_label), active_pixels)
                    bkg = self._read_partial_tree(nside, f.Get(bkg_label), active_pixels)

                    this_data_analysis_bin = DataAnalysisBin(name,
                                                             SparseHealpix(counts, active_pixels, nside),
                                                             SparseHealpix(bkg, active_pixels, nside),
                                                             active_pixels_ids=active_pixels,
                                                             scheme='RING')

                else:

                    # Read the entire sky.

                    counts = tree_to_ndarray(f.Get(bin_label), "count").astype(np.float64)
                    bkg = tree_to_ndarray(f.Get(bkg_label), "count").astype(np.float64)

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

        :return: analysis bin_name iterator
        """

        for analysis_bin in self._data_analysis_bins:

            yield analysis_bin

    def __getitem__(self, item):
        """
        This allows to access the analysis bins as:

        first_analysis_bin = maptree[0]

        or by name:

        first_analysis_bin = maptree["bin_name 0"]

        :param item: integer for serial access, string for access by name
        :return: the analysis bin_name
        """

        if isinstance(item, six.string_types):

            try:

                id = self._data_bins_labels.index(item)

            except ValueError:

                raise KeyError("There is no analysis bin_name named %s" % item)

            else:

                return self._data_analysis_bins[id]

        else:

            try:

                return self._data_analysis_bins[item]

            except IndexError:

                raise IndexError("Analysis bin_name with index %i does not exist" % (item))

    def __len__(self):

        return len(self._data_analysis_bins)

    def _get_entry_list(self, nside, elements_to_read):

        # Create TEventList
        entrylist = ROOT.TEntryList()

        # Add the selections
        _ = map(entrylist.Enter, elements_to_read)

        return entrylist

    def _read_partial_tree(self, nside, ttree_instance, elements_to_read):

        # Decide whether to use a smart loading scheme, or just loading the whole thing, based on the
        # number of elements to be read

        if elements_to_read.shape[0] < 500000:

            # Use a smart loading scheme, where we read only the pixels we need

            # The fastest method that I found is to create a TEventList, apply it to the
            # tree, get a copy of the subset and then use ttree2array

            entrylist = self._get_entry_list(nside, elements_to_read)

            # Apply the EntryList to the tree
            ttree_instance.SetEntryList(entrylist)

            # Get copy of the subset
            # We need to create a dumb TFile to silence a lot of warnings from ROOT
            dumb_tfile = ROOT.TFile("__test.root", "RECREATE")
            new_tree = ttree_instance.CopyTree("")

            # Actually read it from disk
            partial_map = root_numpy.tree2array(new_tree, "count").astype(np.float64)

            # Now reset the entry list
            ttree_instance.SetEntryList(0)

            dumb_tfile.Close()
            os.remove("__test.root")

        else:

            # The smart scheme starts to become slower than the brute force approach, so let's read the whole thing
            partial_map = tree_to_ndarray(ttree_instance, "count").astype(np.float64)

            assert partial_map.shape[0] >= elements_to_read.shape[0], "Trying to read more pixels than present in TTree"

            # Unless we have read the whole sky, let's remove the pixels we shouldn't have read

            if elements_to_read.shape[0] != partial_map.shape[0]:

                # Now let's remove the pixels we shouldn't have read
                partial_map = partial_map[elements_to_read]

        return partial_map

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

            sparse_obs = analysis_bin.observation_map.as_partial()
            sparse_bkg = analysis_bin.background_map.as_partial()

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
        print("Total data size: %.2f Mb" % (size * u.byte).to(u.megabyte).value)