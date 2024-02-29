from builtins import str
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hawc_hal.healpix_handling.sparse_healpix import DenseHealpix, SparseHealpix

ndarray: TypeAlias = NDArray[np.float64]


class DataAnalysisBin:
    """Return a container with data and background maps for a given bin

    :param name: analysi bin name id from maptree
    :param observation_hpx_map: data HEALPix map (partial or dense)
    :param background_hpx_map: background HEALPix map (partial or dense)
    :param active_pixels_ids: HEALPix pixel ids that are within the selected ROI
    :param n_transits: Number of transits for the observation
    :param scheme: HEALPix scheme, defaults to "RING"
    """

    def __init__(
        self,
        name: str,
        observation_hpx_map: SparseHealpix | DenseHealpix,
        background_hpx_map: SparseHealpix | DenseHealpix,
        active_pixels_ids: ndarray,
        n_transits: float | int,
        scheme="RING",
    ) -> None:
        # Get nside
        self._nside = observation_hpx_map.nside

        nside_bkg = background_hpx_map.nside

        assert self._nside == nside_bkg, (
            "Observation and background maps have "
            "different nside (%i vs %i)" % (self._nside, nside_bkg)
        )

        self._npix = observation_hpx_map.npix

        # Store healpix maps
        self._observation_hpx_map = observation_hpx_map

        self._background_hpx_map = background_hpx_map

        # Store the active pixels (i.e., the pixels that are within the selected ROI)
        self._active_pixels_ids = active_pixels_ids

        # Store name and scheme
        self._name = str(name)

        assert scheme in ["RING", "NEST"], "Scheme must be either RING or NEST"

        self._scheme = scheme

        self._n_transits = n_transits

    def to_pandas(self) -> tuple[pd.DataFrame, dict[str, int | float]]:
        # Make a dataframe
        df = pd.DataFrame.from_dict(
            {
                "observation": self._observation_hpx_map.to_pandas(),
                "background": self._background_hpx_map.to_pandas(),
            }
        )

        if self._active_pixels_ids is not None:
            # We are saving only a subset
            df.set_index(self._active_pixels_ids, inplace=True)

        # Some metadata
        meta = {
            "scheme": 0 if self._scheme == "RING" else 1,
            "n_transits": self._n_transits,
            "nside": self._nside,
        }

        return df, meta

    @property
    def name(self) -> str:
        """Return analysis bin name"""
        return self._name

    @property
    def n_transits(self) -> float:
        """Get the number of transits for the observation"""
        return self._n_transits

    @property
    def scheme(self) -> str:
        """Get the current HEALPix scheme (RING or NEST)"""
        return self._scheme

    @property
    def nside(self) -> int:
        """Get the current HEALPix nside"""
        return self._nside

    @property
    def npix(self) -> int:
        """Get the number of active pixels"""
        return self._npix

    @property
    def observation_map(self) -> SparseHealpix | DenseHealpix:
        """Return observation HEALPix map"""
        return self._observation_hpx_map

    @property
    def background_map(self) -> SparseHealpix | DenseHealpix:
        """Return background HEALPix map"""
        return self._background_hpx_map
