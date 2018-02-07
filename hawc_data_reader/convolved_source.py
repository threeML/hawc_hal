import numpy as np
import healpy as hp

from special_values import UNSEEN
from sparse_healpix import SparseHealpix


class ConvolvedPointSource(object):

    def __init__(self, pts_id, likelihood_model, response, maptree):

        self._likelihood_model = likelihood_model

        # Get position
        ra, dec = self._likelihood_model.get_point_source_position(pts_id)

        # Get name
        self._name = self._likelihood_model.get_point_source_name(pts_id)

        self._response_bins, dec_id = response.get_response_dec_bin(dec)

        #print("Using dec bin %s for source %s" % (dec_id, self._name))

        self._pts_id = pts_id

        # Compute the maps
        self._source_maps = []

        for response_bin, data_analysis_bin in zip(self._response_bins, maptree):

            this_psf = response_bin.psf
            this_nside = data_analysis_bin.nside

            # Align the map before storing it, i.e., evaluate it in exactly the same pixels as the
            # data so that we can compare them directly in the get_log_like method

            active_pixels_ids = data_analysis_bin.active_pixels_ids

            if active_pixels_ids is not None:

                this_map_ = this_psf.point_source_image(this_nside, ra, dec, 'icrs')

                # Put the UNSEEN values within the ROI to 0.0 so that the sparse map will be aligned with
                # the data

                this_map_[active_pixels_ids] = np.where(np.isnan(this_map_[active_pixels_ids]),
                                                        0.0,
                                                        this_map_[active_pixels_ids])

                this_map = this_map_[active_pixels_ids]

                # Explicitly remove reference to the full healpix map so that the garbage collector of Python can
                # free the memory before the end of the loop
                del this_map_

            else:

                # We are dealing with a full-sky analysis, we need to use the dense representation

                this_map = np.array(this_psf.point_source_image(this_nside,
                                                                ra, dec,
                                                                'icrs',
                                                                None))

            # Store the map. We will re-weight it according to the new spectrum in the get_log_like method.
            # NOTE: this_map is at this point a np.array, not a SparseHealpix nor
            # a DenseHealpix. This is so that in the get_log_like method we can deal with it more naturally
            # and faster
            self._source_maps.append(this_map)

    @property
    def name(self):

        return self._name

    @property
    def pts_id(self):
        return self._pts_id

    @property
    def source_maps(self):

        return self._source_maps

    @property
    def response_bins(self):

        return self._response_bins

    def get_expected_signal_per_transit(self, response_bin_id, tag=None):

        response_bin = self._response_bins[response_bin_id]

        # Compute the fluxes from the spectral function at the same energies as the simulated function
        energy_centers_keV = response_bin.sim_energy_bin_centers * 1e9  # astromodels expects energies in keV

        # This call needs to be here because the parameters of the model might change,
        # for example during a fit

        source_diff_spectrum = self._likelihood_model.get_point_source_fluxes(self._pts_id,
                                                                              energy_centers_keV,
                                                                              tag=tag)

        # Transform from keV^-1 cm^-2 s^-1 to TeV^-1 cm^-2 s^-1
        source_diff_spectrum *= 1e9

        # Re-weight the detected counts
        scale = source_diff_spectrum / response_bin.sim_differential_photon_fluxes

        return np.sum(scale * response_bin.sim_signal_events_per_bin)