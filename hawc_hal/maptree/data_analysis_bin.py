from builtins import str
from builtins import object
import pandas as pd


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

    def to_pandas(self):

        # Make a dataframe
        df = pd.DataFrame.from_dict({'observation': self._observation_hpx_map.to_pandas(),
                                     'background': self._background_hpx_map.to_pandas()})

        if self._active_pixels_ids is not None:
            # We are saving only a subset
            df.set_index(self._active_pixels_ids, inplace=True)

        # Some metadata
        meta = {'scheme': 0 if self._scheme == 'RING' else 1,
                'n_transits': self._n_transits,
                'nside': self._nside}

        return df, meta

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
