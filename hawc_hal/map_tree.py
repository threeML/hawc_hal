import numpy as np
import root_numpy
import pandas as pd
import healpy as hp
import six
import os
from serialize import Serialization

import ROOT
ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )


from threeML.io.cern_root_utils.io_utils import open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray
from threeML.io.rich_display import display
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename

from region_of_interest import HealpixROIBase, get_roi_from_dict
from hawc_hal.healpix_handling.sparse_healpix import SparseHealpix, DenseHealpix

import astropy.units as u


class DataAnalysisBin(object):

    def __init__(self, name, observation_hpx_map, background_hpx_map, active_pixels_ids, n_transits, scheme='RING'):

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
        self._name = str(name)

        assert scheme in ['RING', 'NEST'], "Scheme must be either RING or NEST"

        self._scheme = scheme

        self._n_transits = n_transits

    def to_hdf(self, serializer):

        # Make a dataframe
        df = pd.DataFrame.from_items((('observation', self._observation_hpx_map.to_pandas()),
                                      ('background', self._background_hpx_map.to_pandas())))

        if self._active_pixels_ids is not None:
            # We are saving only a subset
            df.set_index(self._active_pixels_ids, inplace=True)

        # Add bin_ in front of the name otherwise pandas will complain that the name is not a valid python identifier,
        # if the name is only numbers

        serializer.store_pandas_object("bin_%s" % self._name,
                                       df,
                                       scheme=self._scheme,
                                       n_transits=self._n_transits,
                                       nside=self._nside)

    @property
    def name(self):

        return self._name

    @property
    def n_transits(self):

        return self._n_transits

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

    # Sanitize files in input (expand variables and so on)
    map_tree_file = sanitize_filename(map_tree_file)

    if os.path.splitext(map_tree_file)[-1] == '.root':

        return MapTree.from_root_file(map_tree_file, roi)

    else:

        return MapTree.from_hdf5(map_tree_file, roi)


class MapTree(object):

    def __init__(self, data_bins_labels, data_analysis_bins, roi):

        self._data_bins_labels = data_bins_labels
        self._data_analysis_bins = data_analysis_bins
        self._roi = roi

    @classmethod
    def from_hdf5(cls, map_tree_file, roi):

        with Serialization(map_tree_file) as serializer:

            # Read all data

            data_analysis_bins = []
            data_bins_labels = []

            # Get the ROI from the file (if any)
            if '/ROI' in serializer.keys:

                _, meta = serializer.retrieve_pandas_object('/ROI')

                file_roi = get_roi_from_dict(meta)

            else:

                # Full sky map tree

                file_roi = None

            # Keep also track the bin with the highest NSIDE (note that normally all bins will have the same, but we
            # might relax this in the future). This will be used later on to check the provided ROI against the file ROI

            highest_nside = 0

            # NOTE: we assume that the bins are written in order in the file
            for key in serializer.keys:

                if key.find("/bin_") == 0:

                    # Remove the "bin_" part
                    label = key[5:]
                    data_bins_labels.append(label)

                    # A data analysis bin
                    df, meta = serializer.retrieve_pandas_object(key)

                    if roi is not None:

                        # Reading a specific ROI
                        active_pixels_user = roi.active_pixels(meta['nside'])

                        if file_roi is not None:
                            # Make sure the ROI provided by the user and the ROI in the file are compatible
                            # (i.e., either they are the same, or the user-provided one is smaller)
                            active_pixels_file = file_roi.active_pixels(meta['nside'])

                            assert set(active_pixels_file) == set(df.index)

                            # This verifies that active_pixels_file is a superset (or equal) to the user-provided set
                            assert set(active_pixels_file) >= set(active_pixels_user), \
                                "The ROI you provided (%s) is not a subset " \
                                "of the one contained in the file (%s)" % (roi, file_roi)

                        # Read only the pixels desired by the user
                        observation_hpx_map = SparseHealpix(df.loc[active_pixels_user, 'observation'].values,
                                                            active_pixels_user, meta['nside'])
                        background_hpx_map = SparseHealpix(df.loc[active_pixels_user, 'background'].values,
                                                           active_pixels_user, meta['nside'])

                    else:

                        # Trying to read the entire sky
                        #
                        observation_hpx_map = DenseHealpix(df.loc[:, 'observation'].values)
                        background_hpx_map = DenseHealpix(df.loc[:, 'background'].values)

                        # This signals the DataAnalysisBin that we are dealing with a full sky map
                        active_pixels_user = None

                    this_bin = DataAnalysisBin(label,
                                               observation_hpx_map=observation_hpx_map,
                                               background_hpx_map=background_hpx_map,
                                               active_pixels_ids=active_pixels_user,
                                               n_transits=meta['n_transits'],
                                               scheme=meta['scheme'])

                    data_analysis_bins.append(this_bin)

                    if this_bin.nside > highest_nside:

                        highest_nside = this_bin.nside

                else:

                    # The only other type of key should be the definition of ROI (which we read already)
                    assert key == '/ROI', "Extraneous key in HDF response (%s)" % key

            return cls(data_bins_labels, data_analysis_bins, roi)

    @classmethod
    def from_root_file(cls, map_tree_file, roi):
        """
        Create a MapTree object from a ROOT file and a ROI. Do not use this directly, use map_tree_factory instead.

        :param map_tree_file:
        :param roi:
        :return:
        """

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

                    counts = cls._read_partial_tree(nside, f.Get(bin_label), active_pixels)
                    bkg = cls._read_partial_tree(nside, f.Get(bkg_label), active_pixels)

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

        return cls(data_bins_labels, data_analysis_bins, roi)

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

    @staticmethod
    def _read_partial_tree(nside, ttree_instance, elements_to_read):

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

        print("This Map Tree contains %.3f transits in the first bin" % self._data_analysis_bins[0].n_transits)
        print("Total data size: %.2f Mb" % (size * u.byte).to(u.megabyte).value)

    def write(self, filename):
        """
        Export the tree to a HDF5 file.

        NOTE: if an ROI has been applied, only the data within the ROI will be saved. Actually, a sparse representation
        of the full map will be saved with only the active pixels having a value different than np.nan.

        :param filename: output filename. Use an extension .hd5 or .hdf5 to ensure proper handling by downstream
        software
        :return: None
        """

        with Serialization(filename) as serializer:

            # Write all data analysis bins
            for analysis_bin in self._data_analysis_bins:

                analysis_bin.to_hdf(serializer)

            # Write the ROI
            if self._roi is not None:

                serializer.store_pandas_object('ROI', pd.Series(), **self._roi.to_dict())


