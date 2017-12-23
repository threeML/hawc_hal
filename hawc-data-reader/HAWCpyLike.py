import collections
import numpy as np
import astropy.units as u

from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial

from map_tree import MapTree
from response import HAWCResponse
from convolved_source import ConvolvedPointSource


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


def log_likelihood(observed_counts, expected_bkg_counts, expected_model_counts):

    log_likes = np.zeros(expected_bkg_counts.shape[0])

    predicted_counts = expected_bkg_counts + expected_model_counts

    # o log(m) is 0 if o=0 and m=0, but for the computer 0 x nan = nan, so we need
    # to divide the cases

    idx = (observed_counts > 0)

    log_likes[idx] = observed_counts[idx] * np.log(predicted_counts[idx]) - predicted_counts[idx]

    log_likes[~idx] = -predicted_counts[~idx]

    # The log(o!) factor is not needed for fitting, but it allows the likelihood to -> chi2 for a high
    # number of counts, so we keep it

    log_likes -= logfactorial(observed_counts)

    return np.sum(log_likes)


class HAWCpyLike(PluginPrototype):

    def __init__(self, name, maptree, response, roi):

        # Read map tree (data)

        self._maptree = MapTree(maptree, roi=roi)

        # Read detector response

        self._response = HAWCResponse(response)

        # Make sure that the response and the map tree are aligned
        assert len(self._maptree) == self._response.n_energy_planes, "Response and map tree are not aligned"

        # No nuisance parameters at the moment

        self._nuisance_parameters = collections.OrderedDict()

        # Instance parent class

        super(HAWCpyLike, self).__init__(name, self._nuisance_parameters)

        self._likelihood_model = None

        # These lists will contain the maps for the point sources
        self._convolved_point_sources = ConvolvedSourcesContainer()

        # By default all energy/nHit bins are used
        self._all_planes = range(len(self._maptree))
        self._active_planes = range(len(self._maptree))

    def set_active_measurements(self, bin_id_min, bin_id_max):

        assert bin_id_min in self._all_planes and bin_id_max in self._all_planes, "Illegal bin numbers"

        self._active_planes = range(bin_id_min, bin_id_max+1)

    def display(self):

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

        # For each point source in the model, build the source map
        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        for pts_id in range(n_point_sources):

            this_convolved_point_source = ConvolvedPointSource(pts_id,
                                                               self._likelihood_model,
                                                               self._response,
                                                               self._maptree)

            self._convolved_point_sources.append(this_convolved_point_source)

        n_extended_sources = self._likelihood_model.get_number_of_extended_sources()

        for ext_id in range(n_extended_sources):

            raise NotImplementedError("Extended sources are not supported yet")

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        # This will hold the total log-likelihood
        total_log_like = 0

        for i, data_analysis_bin in enumerate(self._maptree):

            if i not in self._active_planes:

                continue

            # Compute the expectation from the model

            this_model_map = None

            # Make sure that no source has been added since we filled the cache
            assert n_point_sources == self._convolved_point_sources.n_sources_in_cache, "The number of sources has changed. " \
                                                                                    "Please re-assign the model to " \
                                                                                    "the plugin."

            for pts_id in range(n_point_sources):

                this_convolved_source = self._convolved_point_sources[pts_id]

                expectation_per_transit = this_convolved_source.get_expected_signal_per_transit(i, tag=None)

                expectation = expectation_per_transit * self._maptree.n_transits

                these_maps = this_convolved_source.source_maps

                if this_model_map is None:

                    # First addition

                    this_model_map = expectation * these_maps[i]

                else:

                    this_model_map += expectation * these_maps[i]

            # Now compare with observation
            this_log_like = log_likelihood(data_analysis_bin.observation_map.as_array(cache=True),
                                           data_analysis_bin.background_map.as_array(cache=True),
                                           this_model_map)

            total_log_like += this_log_like

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


