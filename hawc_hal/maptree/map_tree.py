from __future__ import absolute_import, division

import os
from builtins import object

import numpy as np
import pandas as pd
import uproot
from past.utils import old_div
from threeML.io.file_utils import sanitize_filename
from threeML.io.logging import setup_logger
from threeML.io.rich_display import display

log = setup_logger(__name__)
log.propagate = False

import astropy.units as u

from ..serialize import Serialization
from .from_hdf5_file import from_hdf5_file
from .from_root_file import from_root_file


def map_tree_factory(map_tree_file, roi, n_transits=None):

    # Sanitize files in input (expand variables and so on)
    map_tree_file = sanitize_filename(map_tree_file)

    if os.path.splitext(map_tree_file)[-1] == ".root":

        return MapTree.from_root_file(map_tree_file, roi, n_transits)

    else:

        return MapTree.from_hdf5(map_tree_file, roi, n_transits)


class MapTree(object):
    def __init__(self, analysis_bins, roi, n_transits=None):

        self._analysis_bins = analysis_bins
        self._roi = roi

        if n_transits is not None:
            self._n_transits = n_transits
        else:
            # find the transits in the map
            transits = []
            for analysis_bin_key in self._analysis_bins:
                transits.append(self._analysis_bins[analysis_bin_key].n_transits)

            # get the largest value
            self._n_transits = np.max(np.array(transits))

    @classmethod
    # def from_hdf5(cls, map_tree_file, roi):
    def from_hdf5(cls, map_tree_file, roi, n_transits):

        data_analysis_bins, transits = from_hdf5_file(map_tree_file, roi, n_transits)

        return cls(data_analysis_bins, roi, transits)

    @classmethod
    def from_root_file(cls, map_tree_file, roi, n_transits):
        """
        Create a MapTree object from a ROOT file and a ROI. Do not use this directly, use map_tree_factory instead.

        :param map_tree_file:
        :param roi:
        :return:
        """

        data_analysis_bins, transits = from_root_file(map_tree_file, roi, n_transits)

        return cls(data_analysis_bins, roi, transits)

    def __iter__(self):
        """
        This allows to loop over the analysis bins as in:

        for analysis_bin in maptree:

            ... do something ...

        :return: analysis bin_name iterator
        """

        for analysis_bin in self._analysis_bins:

            yield analysis_bin

    def __getitem__(self, item):
        """
        This allows to access the analysis bins by name:

        first_analysis_bin = maptree["bin_name 0"]

        :param item: string for access by name
        :return: the analysis bin_name
        """

        try:

            return self._analysis_bins[item]

        except IndexError:

            raise IndexError("Analysis bin_name with index %i does not exist" % (item))

    def __len__(self):

        return len(self._analysis_bins)

    @property
    def n_transits(self):
        return self._n_transits

    @property
    def analysis_bins_labels(self):

        return list(self._analysis_bins.keys())

    def display(self):

        df = pd.DataFrame()

        df["Bin"] = list(self._analysis_bins.keys())
        df["Nside"] = [self._analysis_bins[bin_id].nside for bin_id in self._analysis_bins]
        df["Scheme"] = [self._analysis_bins[bin_id].scheme for bin_id in self._analysis_bins]

        # Compute observed counts, background counts, how many pixels we have in the ROI and
        # the sky area they cover
        n_bins = len(self._analysis_bins)

        obs_counts = np.zeros(n_bins)
        bkg_counts = np.zeros_like(obs_counts)
        n_pixels = np.zeros(n_bins, dtype=int)
        sky_area = np.zeros_like(obs_counts)

        size = 0

        for i, bin_id in enumerate(self._analysis_bins):

            analysis_bin = self._analysis_bins[bin_id]

            sparse_obs = analysis_bin.observation_map.as_partial()
            sparse_bkg = analysis_bin.background_map.as_partial()

            size += sparse_obs.nbytes
            size += sparse_bkg.nbytes

            obs_counts[i] = sparse_obs.sum()
            bkg_counts[i] = sparse_bkg.sum()
            n_pixels[i] = sparse_obs.shape[0]
            sky_area[i] = n_pixels[i] * analysis_bin.observation_map.pixel_area

        df["Obs counts"] = obs_counts
        df["Bkg counts"] = bkg_counts
        df["obs/bkg"] = old_div(obs_counts, bkg_counts)
        df["Pixels in ROI"] = n_pixels
        df["Area (deg^2)"] = sky_area

        display(df)

        first_bin_id = list(self._analysis_bins.keys())[0]
        log.info(
            "This Map Tree contains %.3f transits in the first bin"
            % self._analysis_bins[first_bin_id].n_transits
        )
        log.info("Total data size: %.2f Mb" % (size * u.byte).to(u.megabyte).value)

    def write(self, filename):
        """
        Export the tree to a HDF5 file.

        NOTE: if an ROI has been applied, only the data within the ROI will be saved.

        :param filename: output filename. Use an extension .hd5 or .hdf5 to ensure proper handling by downstream
        software
        :return: None
        """

        # Make a dataframe with the ordered list of bin names
        # bin_names = map(lambda x:x.name, self._analysis_bins)

        # Create a dataframe with a multi-index, with the energy bin name as first level and the HEALPIX pixel ID
        # as the second level
        multi_index_keys = []
        dfs = []
        all_metas = []

        for bin_id in self._analysis_bins:

            analysis_bin = self._analysis_bins[bin_id]

            assert bin_id == analysis_bin.name, "Bin name inconsistency: {} != {}".format(
                bin_id, analysis_bin.name
            )

            multi_index_keys.append(analysis_bin.name)

            this_df, this_meta = analysis_bin.to_pandas()

            dfs.append(this_df)
            all_metas.append(pd.Series(this_meta))

        analysis_bins_df = pd.concat(dfs, axis=0, keys=multi_index_keys)
        meta_df = pd.concat(all_metas, axis=1, keys=multi_index_keys).T

        with Serialization(filename, mode="w") as serializer:

            serializer.store_pandas_object("/analysis_bins", analysis_bins_df)
            serializer.store_pandas_object("/analysis_bins_meta", meta_df)

            # Write the ROI
            if self._roi is not None:

                if self._roi.to_dict()["ROI type"] == "HealpixMapROI":
                    ROIDict = self._roi.to_dict()
                    roimap = ROIDict["roimap"]
                    ROIDict.pop("roimap", None)
                    serializer.store_pandas_object("/ROI", pd.Series(roimap), **ROIDict)

                else:

                    serializer.store_pandas_object("/ROI", pd.Series(), **self._roi.to_dict())

            else:

                serializer.store_pandas_object("/ROI", pd.Series())
