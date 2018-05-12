import collections
import numpy as np
import healpy as hp
import astropy.units as u
from numba import jit, float64
import matplotlib.pyplot as plt

from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial

from map_tree import map_tree_factory
from response import hawc_response_factory
from convolved_source import ConvolvedPointSource, ConvolvedExtendedSource3D, ConvolvedExtendedSource2D
from partial_image_to_healpix import FlatSkyToHealpixTransform
from sparse_healpix import SparseHealpix
from psf_fast import PSFConvolutor


class ConvolvedSourcesContainer(object):
    def __init__(self):

        self._cache = []

    def reset(self):

        self._cache = []

    def __getitem__(self, item):

        return self._cache[item]

    def append(self, convolved_point_source):

        self._cache.append(convolved_point_source)

    @property
    def n_sources_in_cache(self):

        return len(self._cache)

    @property
    def size(self):

        size = 0 * u.byte

        for convolved_point_source in self._cache:

            for point_source_map in convolved_point_source.source_maps:
                size += point_source_map.nbytes * u.byte

        return size.to(u.megabyte)


# This function has two signatures in numba because if there are no sources in the likelihood model,
# then expected_model_counts is 0.0
@jit(["float64(float64[:], float64[:], float64[:])", "float64(float64[:], float64[:], float64)"],
     nopython=True, parallel=False)
def log_likelihood(observed_counts, expected_bkg_counts, expected_model_counts):
    """
    Poisson log-likelihood minus log factorial minus bias. The bias migth be needed to keep the numerical value
    of the likelihood small enough so that there aren't numerical problems when computing differences between two
    likelihood values.

    :param observed_counts:
    :param expected_bkg_counts:
    :param expected_model_counts:
    :param bias:
    :return:
    """

    predicted_counts = expected_bkg_counts + expected_model_counts

    # Remember: because of how the DataAnalysisBin in map_tree.py initializes the maps,
    # observed_counts > 0 everywhere

    log_likes = observed_counts * np.log(predicted_counts) - predicted_counts

    return np.sum(log_likes)


class HAWCpyLike(PluginPrototype):

    # NOTE: the pre-defined flat_sky_pixels_sizes are the truncation radius for the PSF at the Crab position
    # divided by 100

    def __init__(self, name, maptree, response_file, roi, flat_sky_pixels_sizes=0.17):

        # Store ROI
        self._roi = roi

        # Read map tree (data)

        self._maptree = map_tree_factory(maptree, roi=roi)

        # Read detector response_file

        self._response = hawc_response_factory(response_file)

        # Make sure that the response_file and the map tree are aligned
        assert len(self._maptree) == self._response.n_energy_planes, "Response and map tree are not aligned"

        # No nuisance parameters at the moment

        self._nuisance_parameters = collections.OrderedDict()

        # Instance parent class

        super(HAWCpyLike, self).__init__(name, self._nuisance_parameters)

        self._likelihood_model = None

        # These lists will contain the maps for the point sources
        self._convolved_point_sources = ConvolvedSourcesContainer()
        # and this one for extended sources
        self._convolved_ext_sources = ConvolvedSourcesContainer()

        # By default all energy/nHit bins are used
        self._all_planes = range(len(self._maptree))
        self._active_planes = range(len(self._maptree))

        # Set up the flat-sky projection

        self._flat_sky_projection = roi.get_flat_sky_projection(flat_sky_pixels_sizes)

        # Set up the transformations from the flat-sky projection to Healpix, as well as the list of active pixels
        # (one for each energy/nHit bin). We make a separate transformation because different energy bins might have
        # different nsides
        self._active_pixels = []
        self._flat_sky_to_healpix_transform = []

        for i, this_maptree in enumerate(self._maptree):

            this_nside = this_maptree.nside
            this_active_pixels = roi.active_pixels(this_nside)

            this_flat_sky_to_hpx_transform = FlatSkyToHealpixTransform(self._flat_sky_projection.wcs,
                                                                       'icrs',
                                                                       this_nside,
                                                                       this_active_pixels,
                                                                       order='bilinear')

            self._active_pixels.append(this_active_pixels)
            self._flat_sky_to_healpix_transform.append(this_flat_sky_to_hpx_transform)

        # Setup the PSF convolutors for Extended Sources

        self._central_response_bins, dec_bin_id = self._response.get_response_dec_bin(self._roi.ra_dec_center[1])

        print("Using PSF from Dec Bin %i for source %s" % (dec_bin_id, self._name))

        self._psf_convolutors = map(lambda response_bin: PSFConvolutor(response_bin.psf, self._flat_sky_projection),
                                    self._central_response_bins)

        # Pre-compute the log-factorial factor in the likelihood, so we do not keep to computing it over and over
        # again.
        self._log_factorials = np.zeros(len(self._maptree))

        # We also apply a bias so that the numerical value of the log-likelihood stays small. This helps when
        # fitting with algorithms like MINUIT because the convergence criterium involves the difference between
        # two likelihood values, which would be affected by numerical precision errors if the two values are
        # too large
        self._saturated_model_like_per_maptree = np.zeros(len(self._maptree))

        for i, data_analysis_bin in enumerate(self._maptree):

            this_log_factorial = np.sum(logfactorial(data_analysis_bin.observation_map.as_partial()))
            self._log_factorials[i] = this_log_factorial

            # As bias we use the likelihood value for the saturated model
            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            sat_model = np.maximum(obs - bkg, 1e-30).astype(np.float64)

            self._saturated_model_like_per_maptree[i] = log_likelihood(obs, bkg, sat_model) - this_log_factorial

    def get_saturated_model_likelihood(self):
        """
        Returns the likelihood for the saturated model (i.e. a model exactly equal to observation - background).

        :return:
        """
        return np.sum(self._saturated_model_like_per_maptree)

    def set_active_measurements(self, bin_id_min, bin_id_max):

        assert bin_id_min in self._all_planes and bin_id_max in self._all_planes, "Illegal bin_name numbers"

        self._active_planes = range(bin_id_min, bin_id_max + 1)

    def display(self):

        print("Region of Interest: ")
        print("--------------------\n")
        self._roi.display()

        print("")
        print("Flat sky projection: ")
        print("----------------------\n")

        print("Width x height: %s x %s px" % (self._flat_sky_projection.npix_width,
                                              self._flat_sky_projection.npix_height))
        print("Pixel sizes: %s deg" % self._flat_sky_projection.pixel_size)

        print("")
        print("Response: ")
        print("----------\n")

        self._response.display()

        print("")
        print("Map Tree: ")
        print("----------\n")

        self._maptree.display()

        print("")
        print("Active energy/nHit planes: ")
        print("---------------------------\n")
        print(self._active_planes)

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._likelihood_model = likelihood_model_instance

        # Reset
        self._convolved_point_sources.reset()
        self._convolved_ext_sources.reset()

        # For each point source in the model, build the convolution class

        for source in self._likelihood_model.point_sources.values():

            this_convolved_point_source = ConvolvedPointSource(source, self._response, self._flat_sky_projection)

            self._convolved_point_sources.append(this_convolved_point_source)

        # Samewise for point sources

        for source in self._likelihood_model.extended_sources.values():

            if source.spatial_shape.n_dim == 2:

                this_convolved_ext_source = ConvolvedExtendedSource2D(source,
                                                                      self._response,
                                                                      self._flat_sky_projection)

            else:

                this_convolved_ext_source = ConvolvedExtendedSource3D(source,
                                                                      self._response,
                                                                      self._flat_sky_projection)

            self._convolved_ext_sources.append(this_convolved_ext_source)

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # Make sure that no source has been added since we filled the cache
        assert n_point_sources == self._convolved_point_sources.n_sources_in_cache and \
               n_ext_sources == self._convolved_ext_sources.n_sources_in_cache, \
            "The number of sources has changed. Please re-assign the model to the plugin."

        # This will hold the total log-likelihood
        total_log_like = 0

        for i, data_analysis_bin in enumerate(self._maptree):

            if i not in self._active_planes:
                continue

            this_model_map_hpx = self._get_expectation(data_analysis_bin, i, n_point_sources, n_ext_sources)

            # Now compare with observation
            this_pseudo_log_like = log_likelihood(data_analysis_bin.observation_map.as_partial(),
                                                  data_analysis_bin.background_map.as_partial(),
                                                  this_model_map_hpx)

            total_log_like += this_pseudo_log_like - self._log_factorials[i] - self._saturated_model_like_per_maptree[i]

        return total_log_like

    def _get_expectation(self, data_analysis_bin, energy_bin_id, n_point_sources, n_ext_sources):

        # Compute the expectation from the model

        this_model_map = None

        for pts_id in range(n_point_sources):

            this_convolved_source = self._convolved_point_sources[pts_id]

            expectation_per_transit = this_convolved_source.get_source_map(energy_bin_id, tag=None)

            expectation_from_this_source = expectation_per_transit * self._maptree.n_transits

            if this_model_map is None:

                # First addition

                this_model_map = expectation_from_this_source

            else:

                this_model_map += expectation_from_this_source

        # Now process extended sources
        if n_ext_sources > 0:

            this_ext_model_map = None

            for ext_id in range(n_ext_sources):

                this_convolved_source = self._convolved_ext_sources[ext_id]

                expectation_per_transit = this_convolved_source.get_source_map(energy_bin_id)

                if this_ext_model_map is None:

                    # First addition

                    this_ext_model_map = expectation_per_transit

                else:

                    this_ext_model_map += expectation_per_transit

            # Now convolve with the PSF
            if this_model_map is None:
                
                # Only extended sources
            
                this_model_map = (self._psf_convolutors[energy_bin_id].extended_source_image(this_ext_model_map) *
                                  self._maptree.n_transits)
            
            else:

                this_model_map += (self._psf_convolutors[energy_bin_id].extended_source_image(this_ext_model_map) *
                                   self._maptree.n_transits)


        # Now transform from the flat sky projection to HEALPiX

        if this_model_map is not None:

            # First divide for the pixel area because we need to interpolate brightness
            this_model_map = this_model_map / self._flat_sky_projection.project_plane_pixel_area

            this_model_map_hpx = self._flat_sky_to_healpix_transform[energy_bin_id](this_model_map, fill_value=0.0)

            # Now multiply by the pixel area of the new map to go back to flux
            this_model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)

        else:

            # No sources

            this_model_map_hpx = 0.0

        return this_model_map_hpx

    def display_fit(self):

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # This is the resolution (i.e., the size of one pixel) of the image in arcmin
        resolution = 3.0

        # The image is going to cover the diameter plus 20% padding
        xsize = 2.2 * self._roi.data_radius.to("deg").value / (resolution / 60.0)

        n_active_planes = len(self._active_planes)

        fig, subs = plt.subplots(n_active_planes, 2, figsize=(10, n_active_planes * 5))

        active_planes_bins = map(lambda x:self._maptree[x], self._active_planes)

        for i, data_analysis_bin in enumerate(active_planes_bins):

            # Get the center of the projection for this plane
            this_ra, this_dec = self._roi.ra_dec_center

            this_model_map_hpx = self._get_expectation(data_analysis_bin, i, n_point_sources, n_ext_sources)

            # Make a full healpix map for a second
            whole_map = SparseHealpix(this_model_map_hpx,
                                      self._active_pixels[i],
                                      data_analysis_bin.observation_map.nside).as_dense()

            # Add background (which is in a way "part of the model" since the uncertainties are neglected)
            background_map = data_analysis_bin.background_map.as_dense()
            whole_map += background_map

            # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
            if this_ra > 180.0:

                longitude = -180 + (this_ra - 180.0)

            else:

                longitude = this_ra

            # Declination is already between -90 and 90
            latitude = this_dec

            # Select right subplot
            plt.axes(subs[i][0])

            # Plot model
            hp.gnomview(whole_map,
                        title='Energy bin %i' % (i),
                        rot=(longitude, latitude, 0.0),
                        xsize=xsize,
                        reso=resolution,
                        hold=True,
                        notext=True)

            # Select right subplot
            plt.axes(subs[i][1])

            # Plot data

            bkg_subtracted = data_analysis_bin.observation_map.as_dense() - background_map
            idx = np.isnan(bkg_subtracted)
            bkg_subtracted[idx] = hp.UNSEEN

            bkg_subtracted = hp.smoothing(bkg_subtracted,
                                          fwhm=np.deg2rad(1.0),
                                          lmax=100.0)

            hp.gnomview(bkg_subtracted,
                        title='Energy bin %i' % (i),
                        rot=(longitude, latitude, 0.0),
                        xsize=xsize,
                        reso=resolution,
                        hold=True,
                        notext=True)

        return fig

    def display_stacked_image(self, smoothing=1.0):

        # This is the resolution (i.e., the size of one pixel) of the image in arcmin
        resolution = 3.0

        # The image is going to cover the diameter plus 20% padding
        xsize = 2.2 * self._roi.data_radius.to("deg").value / (resolution / 60.0)

        active_planes_bins = map(lambda x: self._maptree[x], self._active_planes)

        # Get the center of the projection for this plane
        this_ra, this_dec = self._roi.ra_dec_center

        # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
        if this_ra > 180.0:

            longitude = -180 + (this_ra - 180.0)

        else:

            longitude = this_ra

        # Declination is already between -90 and 90
        latitude = this_dec

        total = None

        for i, data_analysis_bin in enumerate(active_planes_bins):

            # Plot data
            background_map = data_analysis_bin.background_map.as_dense()
            this_data = data_analysis_bin.observation_map.as_dense() - background_map
            idx = np.isnan(this_data)
            this_data[idx] = hp.UNSEEN

            if i == 0:

                total = this_data

            else:

                # Sum only when there is no UNSEEN, so that the UNSEEN pixels will stay UNSEEN
                total[~idx] += this_data[~idx]

        total_smooth = hp.smoothing(total,
                                    fwhm=np.deg2rad(smoothing),
                                    lmax=100.0,
                                    verbose=False)

        delta_coord = (self._roi.data_radius.to("deg").value * 2.0) / 15.0

        fig, subs = plt.subplots(1, 1, figsize=(5, 5))
        plt.axes(subs)

        hp.gnomview(total_smooth,
                    title='Data',
                    rot=(longitude, latitude, 0.0),
                    xsize=xsize,
                    reso=resolution,
                    hold=True,
                    notext=True)

        hp.graticule(delta_coord, delta_coord)

        return fig

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()

    def get_number_of_data_points(self):

        n_points = 0

        for i, data_analysis_bin in enumerate(self._maptree):
            n_points += data_analysis_bin.observation_map.as_partial().shape[0]

        return n_points
