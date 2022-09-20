from __future__ import absolute_import

import collections
from curses import meta

from threeML.io.logging import setup_logger

from hawc_hal.region_of_interest import get_roi_from_dict
from hawc_hal.serialize import Serialization

log = setup_logger(__name__)
log.propagate = False

from ..healpix_handling import DenseHealpix, SparseHealpix
from .data_analysis_bin import DataAnalysisBin


def from_hdf5_file(map_tree_file, roi, transits):
    """
    Create a MapTree object from a HDF5 file and a ROI. Do not use this directly, use map_tree_factory instead.

    :param map_tree_file:
    :param roi:
    :return:
    """

    # Read the data frames contained in the file
    with Serialization(map_tree_file) as serializer:

        analysis_bins_df, _ = serializer.retrieve_pandas_object("/analysis_bins")
        meta_df, _ = serializer.retrieve_pandas_object("/analysis_bins_meta")
        roimap, roi_meta = serializer.retrieve_pandas_object("/ROI")

    if len(roimap) > 0:
        roi_meta["roimap"] = roimap.values

    # Let's see if the file contains the definition of an ROI
    if len(roi_meta) > 0:

        # Yes. Let's build it
        file_roi = get_roi_from_dict(roi_meta)

        # Now let's check that the ROI the user has provided (if any) is compatible with the one contained
        # in the file (i.e., either they are the same, or the user-provided one is smaller)
        if roi is not None:

            # Let's test with a nside=1024 (the highest we will use in practice)
            active_pixels_file = file_roi.active_pixels(1024)
            active_pixels_user = roi.active_pixels(1024)

            # This verifies that active_pixels_file is a superset (or equal) to the user-provided set
            assert set(active_pixels_file) >= set(active_pixels_user), (
                "The ROI you provided (%s) is not a subset "
                "of the one contained in the file (%s)" % (roi, file_roi)
            )

        else:

            # The user has provided no ROI, but the file contains one. Let's issue a warning
            log.warning(
                "You did not provide any ROI but the map tree %s contains "
                "only data within the ROI %s. "
                "Only those will be used." % (map_tree_file, file_roi)
            )

            # Make a copy of the file ROI and use it as if the user provided that one
            roi = get_roi_from_dict(file_roi.to_dict())

    # Get the name of the analysis bins

    bin_names = analysis_bins_df.index.levels[0]

    # Loop over them and build a DataAnalysisBin instance for each one

    data_analysis_bins = collections.OrderedDict()

    n_transits = meta_df["n_transits"].max() if transits is None else transits

    assert (
        n_transits <= meta_df["transits"].max()
    ), "Cannot use higher value than that of maptree"

    for bin_name in bin_names:

        this_df = analysis_bins_df.loc[bin_name]
        this_meta = meta_df.loc[bin_name]

        if roi is not None:

            # Get the active pixels for this plane
            active_pixels_user = roi.active_pixels(int(this_meta["nside"]))

            # Read only the pixels that the user wants
            observation_hpx_map = SparseHealpix(
                this_df.loc[active_pixels_user, "observation"].values
                * (n_transits / meta_df["n_transits"].max()),
                active_pixels_user,
                this_meta["nside"],
            )
            background_hpx_map = SparseHealpix(
                this_df.loc[active_pixels_user, "background"].values
                * (n_transits / meta_df["n_transits"].max()),
                active_pixels_user,
                this_meta["nside"],
            )

        else:

            # Full sky
            observation_hpx_map = DenseHealpix(
                this_df.loc[:, "observation"].values
                * (n_transits / meta_df["n_transits"].max())
            )
            background_hpx_map = DenseHealpix(
                this_df.loc[:, "background"].values
                * (n_transits / meta_df["n_transits"].max())
            )

            # This signals the DataAnalysisBin that we are dealing with a full sky map
            active_pixels_user = None

        # Let's now build the instance
        this_bin = DataAnalysisBin(
            bin_name,
            observation_hpx_map=observation_hpx_map,
            background_hpx_map=background_hpx_map,
            active_pixels_ids=active_pixels_user,
            # n_transits=this_meta["n_transits"],
            n_transits=n_transits,
            scheme="RING" if this_meta["scheme"] == 0 else "NEST",
        )

        data_analysis_bins[bin_name] = this_bin

    return data_analysis_bins, n_transits
