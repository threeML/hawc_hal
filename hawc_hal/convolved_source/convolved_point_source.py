from __future__ import division
from builtins import zip
from builtins import object
from past.utils import old_div
import os
import collections
import numpy as np

from astromodels import PointSource
from ..psf_fast import PSFInterpolator
from ..interpolation.log_log_interpolation import LogLogInterpolator

from threeML.io.logging import setup_logger
log = setup_logger(__name__)
log.propagate = False


class ConvolvedPointSource(object):

    def __init__(self, source, response, flat_sky_projection):

        assert isinstance(source, PointSource)

        self._source = source

        # Get name
        self._name = self._source.name

        self._response = response

        self._flat_sky_projection = flat_sky_projection

        # This will store the position of the source
        # right now, let's use a fake value so that at the first iteration the source maps will be filled
        # (see get_expected_signal_per_transit)
        self._last_processed_position = (-999, -999)

        self._response_energy_bins = None
        self._psf_interpolators = None

    @property
    def name(self):

        return self._name

    def _update_dec_bins(self, dec_src):

        if abs(dec_src - self._last_processed_position[1]) > 0.1:

            # Source moved by more than 0.1 deg, let's recompute the response and the PSF
            self._response_energy_bins = self._response.get_response_dec_bin(dec_src, interpolate=True)

            # Setup the PSF interpolators
            self._psf_interpolators = collections.OrderedDict()
            for bin_id in self._response_energy_bins:
                self._psf_interpolators[bin_id] = PSFInterpolator(self._response_energy_bins[bin_id].psf,
                                                                  self._flat_sky_projection)

    def get_source_map(self, response_bin_id, tag=None, integrate=False, psf_integration_method='fast'):

        # Get current point source position
        # NOTE: this might change if the point source position is free during the fit,
        # that's why it is here

        ra_src, dec_src = self._source.position.ra.value, self._source.position.dec.value

        if (ra_src, dec_src) != self._last_processed_position:

            # Position changed (or first iteration), let's update the dec bins
            self._update_dec_bins(dec_src)

            self._last_processed_position = (ra_src, dec_src)

        # Get the current response bin
        response_energy_bin = self._response_energy_bins[response_bin_id]
        psf_interpolator = self._psf_interpolators[response_bin_id]

        # Get the PSF image
        # This is cached inside the PSF class, so that if the position doesn't change this line
        # is very fast
        this_map = psf_interpolator.point_source_image(ra_src, dec_src, psf_integration_method)

        # Check that the point source image is entirely contained in the ROI, if not print a warning
        map_sum = this_map.sum()

        if not np.isclose(map_sum, 1.0, rtol=1e-2):

            log.warning("PSF for source %s is not entirely contained "
                                 "in ROI for response bin %s. Fraction is %.2f instead of 1.0. "
                                 "Consider enlarging your model ROI." % (self._name,
                                                                         response_bin_id,
                                                                         map_sum))

        # Compute the fluxes from the spectral function at the same energies as the simulated function
        energy_centers_keV = response_energy_bin.sim_energy_bin_centers * 1e9  # astromodels expects energies in keV

        # This call needs to be here because the parameters of the model might change,
        # for example during a fit

        source_diff_spectrum = self._source(energy_centers_keV, tag=tag)

        # This provide a way to force the use of integration, for testing
        if os.environ.get("HAL_INTEGRATE_POINT_SOURCE", '').lower() == 'yes':  # pragma: no cover

            integrate = True

        if integrate:  # pragma: no cover

            # Slower approach, where we integrate the spectra of both the simulation and the source
            # It is off by default because it is slower and it doesn't provide any improvement in accuracy
            # according to my simulations (GV)

            interp_spectrum = LogLogInterpolator(response_energy_bin.sim_energy_bin_centers,
                                                 source_diff_spectrum * 1e9,
                                                 k=2)

            interp_sim_spectrum = LogLogInterpolator(response_energy_bin.sim_energy_bin_centers,
                                                     response_energy_bin.sim_differential_photon_fluxes,
                                                     k=2)

            src_spectrum = [interp_spectrum.integral(a, b) for (a, b) in zip(response_energy_bin.sim_energy_bin_low,
                                                                             response_energy_bin.sim_energy_bin_hi)]

            sim_spectrum = [interp_sim_spectrum.integral(a, b) for (a, b) in zip(response_energy_bin.sim_energy_bin_low,
                                                                                 response_energy_bin.sim_energy_bin_hi)]

            scale = old_div(np.array(src_spectrum), np.array(sim_spectrum))

        else:

            # Transform from keV^-1 cm^-2 s^-1 to TeV^-1 cm^-2 s^-1 and re-weight the detected counts
            scale = old_div(source_diff_spectrum, response_energy_bin.sim_differential_photon_fluxes) * 1e9

        # Now return the map multiplied by the scale factor
        return np.sum(scale * response_energy_bin.sim_signal_events_per_bin) * this_map
