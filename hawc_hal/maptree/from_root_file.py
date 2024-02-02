from __future__ import absolute_import,annotations

import collections
import multiprocessing
from builtins import str
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import healpy as hp
import numpy as np
import uproot
from numpy.typing import NDArray
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.logging import setup_logger

from hawc_hal.region_of_interest.healpix_cone_roi import HealpixConeROI
from hawc_hal.region_of_interest.healpix_map_roi import HealpixMapROI

from ..healpix_handling import DenseHealpix, SparseHealpix
from ..region_of_interest import HealpixROIBase
from .data_analysis_bin import DataAnalysisBin

log = setup_logger(__name__)
log.propagate = False


@dataclass
class MaptreeMetaData:
    """Metadata for a Maptree file"""

    maptree_ttree_directory: uproot.ReadOnlyDirectory
    _legacy_convention: bool = False

    @property
    def legacy_convention(self) -> bool:
        """Check whether the analysis bin names are prefixed with a zero"""
        nHit_prefix: str = f"nHit{self.analysis_bin_names[0]}"
        if self.maptree_ttree_directory.get(nHit_prefix, None) is None:
            # NOTE: legacy naming scheme
            self._legacy_convention = True

        return self._legacy_convention

    @property
    def analysis_bin_names(self) -> NDArray[np.string_]:
        """Get the analysis bin names contained within the maptree"""
        if self.maptree_ttree_directory.get("BinInfo/name", None) is not None:
            return self.maptree_ttree_directory["BinInfo/name"].array().to_numpy()

        if self.maptree_ttree_directory.get("BinInfo/id", None) is not None:
            return self.maptree_ttree_directory["BinInfo/id"].array().to_numpy()

        raise ValueError("Maptree has an unknown binning scheme convention")

    @property
    def _counts_npixels(self) -> int:
        """Number of pixels within the signal map"""
        bin_id = (
            self.analysis_bin_names[0].zfill(2)
            if self._legacy_convention
            else self.analysis_bin_names[0]
        )
        return self.maptree_ttree_directory[f"nHit{bin_id}/data/count"].member(
            "fEntries"
        )

    @property
    def _bkg_npixels(self) -> int:
        """Number of pixels within the background map"""
        bin_id = (
            self.analysis_bin_names[0].zfill(2)
            if self._legacy_convention
            else self.analysis_bin_names[0]
        )
        return self.maptree_ttree_directory[f"nHit{bin_id}/bkg/count"].member(
            "fEntries"
        )

    @property
    def nside_cnt(self) -> int:
        """Healpix Nside value for the counts map"""
        return hp.pixelfunc.npix2nside(self._counts_npixels)

    @property
    def nside_bkg(self) -> int:
        """Healpix Nside value for the bkg map"""
        return hp.pixelfunc.npix2nside(self._bkg_npixels)

    @property
    def ndurations(self) -> NDArray[np.float64]:
        """Total duration of all bins within the maptree"""
        return self.maptree_ttree_directory["BinInfo/totalDuration"].array().to_numpy()


def get_array_from_file(
    legacy_convention: bool,
    bin_id: str,
    map_infile: uproot.ReadOnlyDirectory,
    hpx_map: NDArray[np.float64],
    roi: Optional[HealpixConeROI | HealpixMapROI] = None,
) -> tuple[str, NDArray[np.float64]]:
    """Load the signal array from a ROOT maptree

    :param legacy_convention: True if there is a zero prefix in the analysis bin name
    :type legacy_convention: bool
    :param bin_id: Analysis bin from the maptree file
    :type bin_id: str
    :param map_infile: Uproot object that handles the reading of the maptree file
    :type map_infile: uproot.ReadOnlyDirectory
    :param hpx_map: Healpix map array that specifies the active pixels within the ROI
    :type hpx_map: NDArray[np.float64]
    :param roi: ROI instance specyfing the region of interest only load a partial
    segment of the map. If set to None, it loads the full sky map (more memory intensive)
    :type roi: Optional[HealpixConeROI | HealpixMapROI], optional
    :return: Returns teh active analysis bin with its corresponding signal array.
    :rtype: tuple[str, NDArray[np.float64]]
    """

    current_bin_id = bin_id.zfill(2) if legacy_convention else bin_id

    if roi is not None:
        # NOTE: load only the pixels within the ROI
        return bin_id, map_infile[
            f"nHit{current_bin_id}/data/count"
        ].array().to_numpy()[hpx_map > 0.0]

    return bin_id, map_infile[f"nHit{current_bin_id}/data/count"].array().to_numpy()


def get_bkg_array_from_file(
    legacy_convention: bool,
    bin_id: str,
    map_infile: uproot.ReadOnlyDirectory,
    hpx_map: NDArray[np.float64],
    roi: Optional[HealpixConeROI | HealpixMapROI] = None,
) -> tuple[str, NDArray[np.float64]]:
    """Load the background array from a ROOT maptree file

    :param legacy_convention: boolean to check if there is a zero prefix
    in the analysis bin name
    :type legacy_convention: bool
    :param bin_id: Analysis bin from the maptree file
    :type bin_id: str
    :param map_infile: uproot.ReadOnlyDirectory object that handles the
    reading of the maptree file
    :type map_infile: uproot.ReadOnlyDirectory
    :param hpx_map: Healpix map array that specifies the active pixels within the ROI
    :type hpx_map: NDArray[np.float64]
    :param roi: ROI object specifying whether there is an active ROI if None,
    then the whole sky is loaded, by default None
    :type roi: Optional[HealpixConeROI | HealpixMapROI], optional
    :return: Returns the active analysis bin with its corresponding background array
    """
    current_bin_id = bin_id.zfill(2) if legacy_convention else bin_id
    if roi is not None:
        # NOTE: load only the pixels within the ROI
        return bin_id, map_infile[f"nHit{current_bin_id}/bkg/count"].array().to_numpy()[
            hpx_map > 0.0
        ]

    return bin_id, map_infile[f"nHit{current_bin_id}/bkg/count"].array().to_numpy()


def from_root_file(
    map_tree_file: Path,
    roi: Union[HealpixConeROI, HealpixMapROI],
    transits: float,
    n_workers: int,
    scheme: int = 0,
):
    """Create a Maptree object from a ROOT file.
    Do not use this directly, use the maptree_factory method instead.

    :param map_tree_file: Maptree ROOT file
    :param roi:  User defined region of interest (ROI)
    :param transits: Number of transits specified within maptree.
    If not specified assume the maximum number of transits for all binss.
    :param n_workers: Numbrer of processes used for parallel reading of ROOT files
    :param scheme: RING or NESTED Healpix scheme (default RING:0), by default 0
    :raises IOError: Raised if file does not exist or is corrupted
    :return: Return dictionary with DataAnalysis objects for the active bins and
    the number of transits
    :rtype: tuple[dict[str, DataAnalysisBin], float]
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

        maptree_metadata = MaptreeMetaData(map_infile)

        maptree_durations = maptree_metadata.ndurations
        legacy_convention = maptree_metadata.legacy_convention
        data_bins_labels = maptree_metadata.analysis_bin_names

        nside_cnt: int = maptree_metadata.nside_cnt
        nside_bkg: int = maptree_metadata.nside_bkg
        # binning_scheme_name: str = maptree_metadata.binning_scheme

        assert (
            nside_cnt == nside_bkg
        ), "Nside value needs to be the same for counts and bkg. maps"

        healpix_map_active = np.zeros(hp.nside2npix(nside_cnt))

        # NOTE: read only the pixels within the ROI
        if roi is not None:
            active_pixels = roi.active_pixels(
                nside_cnt, system="equatorial", ordering="RING"
            )

            healpix_map_active[active_pixels] = 1.0

        signal_data_info = [
            (legacy_convention, name, map_infile, healpix_map_active, roi)
            for name in data_bins_labels
        ]

        # Launch processes to speed up the reading of the maptree file
        # NOTE: The number of workers is suggested to be kept equal one less
        # than the number of available cores in the system.

        with multiprocessing.Pool(processes=n_workers) as executor:
            result_data = list(executor.starmap(get_array_from_file, signal_data_info))
            result_bkg = list(
                executor.starmap(get_bkg_array_from_file, signal_data_info)
            )

        # Processes are not guaranteed to preserve order of analysis bin names
        # Organize them into a dictionary for proper readout
        data_dir_array = dict(result_data)
        bkg_dir_array = dict(result_bkg)

    # The map-maker underestimate the livetime of bins with low statistic
    # by removing time intervals with zero events. Therefore, the best
    # estimate of the livetime is the maximum of n_transits, which normally
    # happen in the bins with high statistic
    max_duration: float = np.divide(maptree_durations.max(), 24.0)

    # use value of maptree unless otherwise specified by user
    n_transits = max_duration if transits is None else transits
    scale_factor: float = n_transits / max_duration

    assert scheme == 0, "NESTED HEALPix is not currently supported."

    data_analysis_bins = collections.OrderedDict()

    if roi is not None:
        for name in data_bins_labels:
            counts = data_dir_array[name]
            bkg = bkg_dir_array[name]

            counts *= scale_factor
            bkg *= scale_factor

            counts_hpx = SparseHealpix(counts, active_pixels, nside_cnt)
            bkg_hpx = SparseHealpix(bkg, active_pixels, nside_bkg)

            # Read only elements within the ROI
            this_data_analysis_bin = DataAnalysisBin(
                name,
                counts_hpx,
                bkg_hpx,
                active_pixels_ids=active_pixels,
                n_transits=n_transits,
                scheme="RING",
            )
            data_analysis_bins[name] = this_data_analysis_bin
    else:
        # If no ROI is provided read the entire sky
        for name in data_bins_labels:
            counts = data_dir_array[name]
            bkg = bkg_dir_array[name]

            counts *= scale_factor
            bkg *= scale_factor

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
