from __future__ import absolute_import, annotations

import collections
from builtins import str
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Union

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
    def analysis_bin_names(self) -> NDArray[np.bytes_]:
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
        return self.maptree_ttree_directory[f"nHit{bin_id}/data/count"].member("fEntries")

    @property
    def _bkg_npixels(self) -> int:
        """Number of pixels within the background map"""
        bin_id = (
            self.analysis_bin_names[0].zfill(2)
            if self._legacy_convention
            else self.analysis_bin_names[0]
        )
        return self.maptree_ttree_directory[f"nHit{bin_id}/bkg/count"].member("fEntries")

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


def _read_data_arrays(
    file_path: str,
    bin_id: str,
    legacy_convention: bool,
    active_pixels: NDArray[np.int64] | None = None,
) -> tuple[str, NDArray[np.float64], NDArray[np.float64]]:
    """Load the signal and background arrays for a single analysis bin.

    Designed to run inside a worker process: each call opens its own
    uproot handle on ``file_path`` so the open ``ReadOnlyDirectory`` does
    not need to cross the process boundary.

    :param file_path: Path to the maptree ROOT file.
    :param bin_id: Analysis bin name from the maptree file.
    :param legacy_convention: True if analysis bin names are zero-padded
        (e.g. ``"01"`` instead of ``"1"``).
    :param active_pixels: Integer indices of the HEALPix pixels inside
        the ROI. If ``None``, the full-sky arrays are returned.
    :return: ``(bin_id, counts, bkg)`` — the bin name plus its signal
        and background arrays, masked to ``active_pixels`` when provided.
    """

    current_bin_id = bin_id.zfill(2) if legacy_convention else bin_id
    with uproot.open(file_path, handler=uproot.MemmapSource) as map_infile:
        # NOTE: load only the pixels within the ROI
        counts = map_infile[f"nHit{current_bin_id}/data/count"].array(library="np")
        bkg = map_infile[f"nHit{current_bin_id}/bkg/count"].array(library="np")
        if active_pixels is not None:
            return bin_id, counts[active_pixels], bkg[active_pixels]

        return bin_id, counts, bkg


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
    """

    map_tree_file = sanitize_filename(map_tree_file)

    # Check that they exists and can be read
    if not file_existing_and_readable(map_tree_file):  # pragma: no cover
        raise IOError(f"MapTree {map_tree_file} does not exist or is not readable")

    # Make sure we have a proper ROI (or None)
    assert isinstance(roi, HealpixROIBase) or roi is None, (
        "You have to provide an ROI choosing from the "
        "available ROIs in the region_of_interest module"
    )

    assert scheme == 0, "NESTED HEALPix is not currently supported."

    if roi is None:
        log.warning("You have set roi=None, so you are reading the entire sky")

    # Read the maptree metadata
    with uproot.open(map_tree_file, handler=uproot.MemmapSource) as map_infile:
        # the handler for MemmapSource loads the file as it's needed
        # suggested as the best for large local files
        # otherwise use MultithreadedFileSource for remote files
        # which requires setting the option for use_threads to True
        log.info("Reading Maptree!")

        maptree_metadata = MaptreeMetaData(map_infile)

        maptree_durations = maptree_metadata.ndurations
        legacy_convention = maptree_metadata.legacy_convention
        data_bins_labels = maptree_metadata.analysis_bin_names

        nside_cnt: int = maptree_metadata.nside_cnt
        nside_bkg: int = maptree_metadata.nside_bkg

        assert nside_cnt == nside_bkg, (
            "Nside value needs to be the same for counts and bkg. maps"
        )

    # NOTE: read only the pixels within the ROI
    if roi is not None:
        active_pixels = roi.active_pixels(nside_cnt, system="equatorial", ordering="RING")
    else:
        active_pixels = None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _read_data_arrays,
                map_tree_file.as_posix(),
                bin_id,
                legacy_convention,
                active_pixels,
            )
            for bin_id in data_bins_labels
        ]

        results = [f.result() for f in futures]

        # Processes are not guaranteed to preserve order of analysis bin names
        # Organize them into a dictionary for proper readout
        data_dir_array = {bin_id: counts for bin_id, counts, _ in results}
        bkg_dir_array = {bin_id: bkg for bin_id, _, bkg in results}

    # The map-maker underestimate the livetime of bins with low statistic
    # by removing time intervals with zero events. Therefore, the best
    # estimate of the livetime is the maximum of n_transits, which normally
    # happen in the bins with high statistic
    max_duration: float = np.divide(maptree_durations.max(), 24.0)

    # use value of maptree unless otherwise specified by user
    n_transits = max_duration if transits is None else transits

    data_analysis_bins = collections.OrderedDict()

    if roi is not None:
        for name in data_bins_labels:
            counts = data_dir_array[name]
            bkg = bkg_dir_array[name]

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
