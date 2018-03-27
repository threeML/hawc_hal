import collections
import numpy as np
import healpy as hp
import astropy.units as u
from numba import jit, float64, float32

from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial

from map_tree import map_tree_factory
from response import hawc_response_factory
from convolved_source import ConvolvedPointSource, ConvolvedExtendedSource2D, ConvolvedExtendedSource3D
from partial_image_to_healpix import FlatSkyToHealpixTransform


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


@jit(float64(float32[:], float32[:], float64[:]), nopython=True, parallel=False)
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

    #(0.15, 0.1497, 0.1098, 0.0928, 0.0675, 0.051, 0.0494, 0.0412, 0.03496, 0.03218)

    def __init__(self, name, maptree, response, roi,
                 flat_sky_pixels_sizes=(0.05, 0.05, 0.05, 0.05, 0.05,
                                        0.05, 0.05, 0.05, 0.05, 0.05)):

        # Store ROI
        self._roi = roi

        # Read map tree (data)

        self._maptree = map_tree_factory(maptree, roi=roi)

        # Read detector response

        self._response = hawc_response_factory(response)

        # Make sure that the response and the map tree are aligned
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

        # Set up the flat-sky projections (one for each energy/nHit bin)
        self._flat_sky_pixels_sizes = flat_sky_pixels_sizes

        self._flat_sky_projections = map(lambda pix_size: roi.get_flat_sky_projection(pix_size),
                                         self._flat_sky_pixels_sizes)

        # Set up the transformations from the flat-sky projections to Healpix, as well as the list of active pixels
        # (one for each energy/nHit bin)
        self._active_pixels = []
        self._flat_sky_to_healpix_transform = []

        for flat_sky_proj, this_maptree in zip(self._flat_sky_projections, self._maptree):

            this_nside = this_maptree.nside
            this_active_pixels = roi.active_pixels(this_nside)
            this_flat_sky_to_hpx_transform = FlatSkyToHealpixTransform(flat_sky_proj.wcs,
                                                                       'icrs',
                                                                       this_nside,
                                                                       this_active_pixels,
                                                                       order='bilinear')

            self._active_pixels.append(this_active_pixels)
            self._flat_sky_to_healpix_transform.append(this_flat_sky_to_hpx_transform)

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
        print("Flat sky projections: ")
        print("----------------------\n")

        widths = []
        heights = []
        pix_sizes = []

        for proj in self._flat_sky_projections:

            widths.append(proj.npix_width)
            heights.append(proj.npix_height)
            pix_sizes.append(proj.pixel_size)

        print("Width x height: %s px" % ", ".join(map(lambda (w, h): "%ix%i" % (w, h), zip(widths, heights))))
        print("Pixel sizes: %s deg" % ", ".join(map(lambda r: "%.3f" % r, pix_sizes)))

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

            this_convolved_point_source = ConvolvedPointSource(source, self._response, self._flat_sky_projections)

            self._convolved_point_sources.append(this_convolved_point_source)

        # Samewise for point sources

        for source in self._likelihood_model.extended_sources.values():

            if source.spatial_shape.n_dim == 2:

                this_convolved_ext_source = ConvolvedExtendedSource2D(source,
                                                                      self._response,
                                                                      self._flat_sky_projections)

            else:

                this_convolved_ext_source = ConvolvedExtendedSource3D(source,
                                                                      self._response,
                                                                      self._flat_sky_projections)

            self._convolved_ext_sources.append(this_convolved_ext_source)

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # This will hold the total log-likelihood
        total_log_like = 0

        for i, data_analysis_bin in enumerate(self._maptree):

            if i not in self._active_planes:
                continue

            # Compute the expectation from the model

            this_model_map = None

            # Make sure that no source has been added since we filled the cache
            assert n_point_sources == self._convolved_point_sources.n_sources_in_cache and \
                   n_ext_sources == self._convolved_ext_sources.n_sources_in_cache, \
                "The number of sources has changed. Please re-assign the model to the plugin."

            for pts_id in range(n_point_sources):

                this_convolved_source = self._convolved_point_sources[pts_id]

                expectation_per_transit = this_convolved_source.get_source_map(i, tag=None)

                expectation_from_this_source = expectation_per_transit * self._maptree.n_transits

                if this_model_map is None:

                    # First addition

                    this_model_map = expectation_from_this_source

                else:

                    this_model_map += expectation_from_this_source

            # Now process extended sources
            for ext_id in range(n_ext_sources):

                this_convolved_source = self._convolved_ext_sources[ext_id]

                expectation_per_transit = this_convolved_source.get_source_map(i)

                expectation_from_this_source = expectation_per_transit * self._maptree.n_transits

                if this_model_map is None:

                    # First addition

                    this_model_map = expectation_from_this_source

                else:

                    this_model_map += expectation_from_this_source

            # Now transform from the flat sky projection to HEALPiX

            # Keep track of the overall normalization
            total_before_interpolation = this_model_map.sum()

            # First divide for the pixel area because we need to interpolate brightness
            this_model_map = this_model_map / self._flat_sky_projections[i].project_plane_pixel_area

            this_model_map_hpx = self._flat_sky_to_healpix_transform[i](this_model_map,
                                                                        fill_value=0.0)

            # Now multiply by the pixel area of the new map to go back to flux
            this_model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)

            # Due to the interpolation the new map might have a slightly different total. The difference
            # is < 1%, but let's fix it anyway
            total_after_interpolation = this_model_map_hpx.sum()

            renorm = total_before_interpolation / total_after_interpolation

            this_model_map_hpx = this_model_map_hpx * renorm

            # Now compare with observation
            this_pseudo_log_like = log_likelihood(data_analysis_bin.observation_map.as_partial(),
                                                  data_analysis_bin.background_map.as_partial(),
                                                  this_model_map_hpx)

            total_log_like += this_pseudo_log_like - self._log_factorials[i] - self._saturated_model_like_per_maptree[i]

        return total_log_like

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
