import numpy as np

from threeML.io.progress_bar import progress_bar

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

        print("Using dec bin %s for source %s" % (dec_id, self._name))

        self._pts_id = pts_id

        with progress_bar(len(maptree), title="Creating point source maps for source %s" % self._name) as p:

            # Compute the maps
            self._source_maps = []

            for response_bin, data_analysis_bin in zip(self._response_bins, maptree):

                this_psf = response_bin.psf
                this_nside = data_analysis_bin.nside

                # Align the map before storing it, i.e., evaluate it in exactly the same pixels as the
                # data so that we can compare them directly in the get_log_like method

                active_pixels_ids = data_analysis_bin.active_pixels_ids

                if active_pixels_ids is not None:

                    # By using the "pixels_to_be_zeroed" keyword we are sure that the output map will be aligned
                    # with the data, i.e., the sparse representation of this map will be aligned with the sparse
                    # representation of the data map, which is needed when comparing them in the get_log_like
                    # method

                    this_map_ = this_psf.point_source_image(this_nside, ra, dec, coordsys='icrs',
                                                            pixels_to_be_zeroed=active_pixels_ids)

                    # Make sure that this map is normalized to 1
                    assert np.isclose(this_map_.as_sparse().sum(), 1.0, rtol=1e-3), \
                        "Source map is not normalized to 1 (total = %s)" % this_map_.sum()

                    # If the sparse data map is smaller than the PSF sparse map in "this_map", we need one more
                    # step to align them
                    # NOTE: the PSF map after this operation might not be normalized to 1 anymore, as some of the
                    # non-null pixels might be lost in the selection if the ROI is not large enough. This is not
                    # an issue, as the map will be renormalized according to the flux of the source.
                    this_map_dense = np.full(data_analysis_bin.npix, fill_value=UNSEEN)

                    this_map_dense[active_pixels_ids] = this_map_.as_sparse()[active_pixels_ids]

                    this_map = np.array(SparseHealpix(this_map_dense, copy=True, fill_value=UNSEEN).as_sparse())

                else:

                    # We are dealing with a full-sky analysis, we need to use the dense representation

                    this_map = np.array(this_psf.point_source_image(this_nside,
                                                                    ra, dec, coordsys='icrs').as_dense())

                # Store the map. We will re-weight it according to the new spectrum in the get_log_like method.
                # NOTE: this_map is at this point a np.array, not a SparseHealpix nor
                # a DenseHealpix. This is so that in the get_log_like method we can deal with it more naturally
                # and faster
                self._source_maps.append(this_map)

                p.increase()

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

        source_diff_spectrum = self._likelihood_model.get_point_source_fluxes(self._pts_id,
                                                                              energy_centers_keV,
                                                                              tag=tag)

        # Transform from keV^-1 cm^-2 s^-1 to TeV^-1 cm^-2 s^-1
        source_diff_spectrum *= 1e9

        # Re-weight the detected counts
        scale = source_diff_spectrum / response_bin.sim_differential_photon_fluxes

        return np.sum(scale * response_bin.sim_signal_events_per_bin)