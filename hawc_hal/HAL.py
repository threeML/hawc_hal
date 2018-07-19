from __future__ import print_function

import copy
import collections

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import poisson

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft as convolve

from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial
from threeML.parallel import parallel_client
from threeML.io.progress_bar import progress_bar

from astromodels import Parameter

from hawc_hal.maptree import map_tree_factory
from hawc_hal.response import hawc_response_factory
from hawc_hal.convolved_source import ConvolvedPointSource, \
    ConvolvedExtendedSource3D, ConvolvedExtendedSource2D, ConvolvedSourcesContainer
from hawc_hal.healpix_handling import FlatSkyToHealpixTransform
from hawc_hal.healpix_handling import SparseHealpix
from hawc_hal.healpix_handling import get_gnomonic_projection
from hawc_hal.psf_fast import PSFConvolutor
from hawc_hal.log_likelihood import log_likelihood
from hawc_hal.util import ra_to_longitude


class HAL(PluginPrototype):
    """
    The HAWC Accelerated Likelihood plugin for 3ML.
    :param name: name for the plugin
    :param maptree: Map Tree (either ROOT or hdf5 format)
    :param response: Response of HAWC (either ROOT or hd5 format)
    :param roi: a ROI instance describing the Region Of Interest
    :param flat_sky_pixels_size: size of the pixel for the flat sky projection (Hammer Aitoff)
    """

    def __init__(self, name, maptree, response_file, roi, flat_sky_pixels_size=0.17):

        # Store ROI
        self._roi = roi

        # Set up the flat-sky projection

        self._flat_sky_projection = roi.get_flat_sky_projection(flat_sky_pixels_size)

        # Read map tree (data)

        self._maptree = map_tree_factory(maptree, roi=roi)

        # Read detector response_file

        self._response = hawc_response_factory(response_file)

        # Use a renormalization of the background as nuisance parameter
        # NOTE: it is fixed to 1.0 unless the user explicitly sets it free (experimental)
        self._nuisance_parameters = collections.OrderedDict()
        self._nuisance_parameters['%s_bkg_renorm' % name] = Parameter('%s_bkg_renorm' % name, 1.0,
                                                                      min_value=0.5, max_value=1.5,
                                                                      delta=0.01,
                                                                      desc="Renormalization for background map",
                                                                      free=False,
                                                                      is_normalization=False)

        # Instance parent class

        super(HAL, self).__init__(name, self._nuisance_parameters)

        self._likelihood_model = None

        # These lists will contain the maps for the point sources
        self._convolved_point_sources = ConvolvedSourcesContainer()
        # and this one for extended sources
        self._convolved_ext_sources = ConvolvedSourcesContainer()

        # All energy/nHit bins are loaded in memory
        self._all_planes = list(self._maptree.analysis_bins_labels)

        # The active planes list always contains the list of *indexes* of the active planes
        self._active_planes = None

        # Set up the transformations from the flat-sky projection to Healpix, as well as the list of active pixels
        # (one for each energy/nHit bin). We make a separate transformation because different energy bins might have
        # different nsides
        self._active_pixels = collections.OrderedDict()
        self._flat_sky_to_healpix_transform = collections.OrderedDict()

        for bin_id in self._maptree:

            this_maptree = self._maptree[bin_id]
            this_nside = this_maptree.nside
            this_active_pixels = roi.active_pixels(this_nside)

            this_flat_sky_to_hpx_transform = FlatSkyToHealpixTransform(self._flat_sky_projection.wcs,
                                                                       'icrs',
                                                                       this_nside,
                                                                       this_active_pixels,
                                                                       (self._flat_sky_projection.npix_width,
                                                                        self._flat_sky_projection.npix_height),
                                                                       order='bilinear')

            self._active_pixels[bin_id] = this_active_pixels
            self._flat_sky_to_healpix_transform[bin_id] = this_flat_sky_to_hpx_transform

        # This will contain a list of PSF convolutors for extended sources, if there is any in the model

        self._psf_convolutors = None

        # Pre-compute the log-factorial factor in the likelihood, so we do not keep to computing it over and over
        # again.
        self._log_factorials = collections.OrderedDict()

        # We also apply a bias so that the numerical value of the log-likelihood stays small. This helps when
        # fitting with algorithms like MINUIT because the convergence criterium involves the difference between
        # two likelihood values, which would be affected by numerical precision errors if the two values are
        # too large
        self._saturated_model_like_per_maptree = collections.OrderedDict()

        # The actual computation is in a method so we can recall it on clone (see the get_simulated_dataset method)
        self._compute_likelihood_biases()

        # This will save a clone of self for simulations
        self._clone = None

        # Integration method for the PSF (see psf_integration_method)
        self._psf_integration_method = "exact"

    @property
    def psf_integration_method(self):
        """
        Get or set the method for the integration of the PSF.

        * "exact" is more accurate but slow, if the position is free to vary it adds a lot of time to the fit. This is
        the default, to be used when the position of point sources are fixed. The computation in that case happens only
        once so the impact on the run time is negligible.
        * "fast" is less accurate (up to an error of few percent in flux) but a lot faster. This should be used when
        the position of the point source is free, because in that case the integration of the PSF happens every time
        the position changes, so several times during the fit.

        If you have a fit with a free position, use "fast". When the position is found, you can fix it, switch to
        "exact" and redo the fit to obtain the most accurate measurement of the flux. For normal sources the difference
        will be small, but for very bright sources it might be up to a few percent (most of the time < 1%). If you are
        interested in the localization contour there is no need to rerun with "exact".

        :param mode: either "exact" or "fast"
        :return: None
        """

        return self._psf_integration_method

    @psf_integration_method.setter
    def psf_integration_method(self, mode):

        assert mode.lower() in ["exact", "fast"], "PSF integration method must be either 'exact' or 'fast'"

        self._psf_integration_method = mode.lower()

    def _setup_psf_convolutors(self):

        central_response_bins = self._response.get_response_dec_bin(self._roi.ra_dec_center[1])

        self._psf_convolutors = collections.OrderedDict()
        for bin_id in central_response_bins:
            self._psf_convolutors[bin_id] = PSFConvolutor(central_response_bins[bin_id].psf, self._flat_sky_projection)

    def _compute_likelihood_biases(self):

        for bin_label in self._maptree:

            data_analysis_bin = self._maptree[bin_label]

            this_log_factorial = np.sum(logfactorial(data_analysis_bin.observation_map.as_partial()))
            self._log_factorials[bin_label] = this_log_factorial

            # As bias we use the likelihood value for the saturated model
            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            sat_model = np.clip(obs - bkg, 1e-50, None).astype(np.float64)

            self._saturated_model_like_per_maptree[bin_label] = log_likelihood(obs, bkg, sat_model) - this_log_factorial

    def get_saturated_model_likelihood(self):
        """
        Returns the likelihood for the saturated model (i.e. a model exactly equal to observation - background).

        :return:
        """
        return sum(self._saturated_model_like_per_maptree.values())

    def set_active_measurements(self, bin_id_min=None, bin_id_max=None, bin_list=None):
        """
        Set the active analysis bins to use during the analysis. It can be used in two ways:

        - Specifying a range: if the response and the maptree allows it, you can specify a minimum id and a maximum id
        number. This only works if the analysis bins are numerical, like in the normal fHit analysis. For example:

            > set_active_measurement(bin_id_min=1, bin_id_max=9(

        - Specifying a list of bins as strings. This is more powerful, as allows to select any bins, even
        non-contiguous bins. For example:

            > set_active_measurement(bin_list=[list])

        :param bin_id_min: minimum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_id_max: maximum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_list: a list of analysis bins to use
        :return: None
        """

        # Check for legal input
        if bin_id_min is not None:

            assert bin_id_max is not None, "If you provide a minimum bin, you also need to provide a maximum bin"

            # Make sure they are integers
            bin_id_min = int(bin_id_min)
            bin_id_max = int(bin_id_max)

            self._active_planes = []
            for this_bin in range(bin_id_min, bin_id_max + 1):
                this_bin = str(this_bin)
                if this_bin not in self._all_planes:

                    raise ValueError("Bin %s it not contained in this response" % this_bin)

                self._active_planes.append(this_bin)

        else:

            assert bin_id_max is None, "If you provide a maximum bin, you also need to provide a minimum bin"

            assert bin_list is not None

            self._active_planes = []

            for this_bin in bin_list:

                if not this_bin in self._all_planes:

                    raise ValueError("Bin %s it not contained in this response" % this_bin)

                self._active_planes.append(this_bin)

    def display(self, verbose=False):
        """
        Prints summary of the current object content.
        """

        print("Region of Interest: ")
        print("-------------------\n")
        self._roi.display()

        print("")
        print("Flat sky projection: ")
        print("--------------------\n")

        print("Width x height: %s x %s px" % (self._flat_sky_projection.npix_width,
                                              self._flat_sky_projection.npix_height))
        print("Pixel sizes: %s deg" % self._flat_sky_projection.pixel_size)

        print("")
        print("Response: ")
        print("---------\n")

        self._response.display(verbose)

        print("")
        print("Map Tree: ")
        print("----------\n")

        self._maptree.display()

        print("")
        print("Active energy/nHit planes ({}):".format(len(self._active_planes)))
        print("-------------------------------\n")
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

        # Samewise for extended sources

        ext_sources = self._likelihood_model.extended_sources.values()

        # NOTE: ext_sources evaluate to False if empty
        if ext_sources:

            # We will need to convolve

            self._setup_psf_convolutors()

            for source in ext_sources:

                if source.spatial_shape.n_dim == 2:

                    this_convolved_ext_source = ConvolvedExtendedSource2D(source,
                                                                          self._response,
                                                                          self._flat_sky_projection)

                else:

                    this_convolved_ext_source = ConvolvedExtendedSource3D(source,
                                                                          self._response,
                                                                          self._flat_sky_projection)

                self._convolved_ext_sources.append(this_convolved_ext_source)

    def display_spectrum(self):
        """
        Make a plot of the current spectrum and its residuals (integrated over space)

        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        total_counts = np.zeros(len(self._active_planes), dtype=float)
        total_model = np.zeros_like(total_counts)
        model_only = np.zeros_like(total_counts)
        net_counts = np.zeros_like(total_counts)
        yerr_low = np.zeros_like(total_counts)
        yerr_high = np.zeros_like(total_counts)

        for i, energy_id in enumerate(self._active_planes):

            data_analysis_bin = self._maptree[energy_id]

            this_model_map_hpx = self._get_expectation(data_analysis_bin, energy_id, n_point_sources, n_ext_sources)

            this_model_tot = np.sum(this_model_map_hpx)

            this_data_tot = np.sum(data_analysis_bin.observation_map.as_partial())
            this_bkg_tot = np.sum(data_analysis_bin.background_map.as_partial())

            total_counts[i] = this_data_tot
            net_counts[i] = this_data_tot - this_bkg_tot
            model_only[i] = this_model_tot

            this_wh_model = this_model_tot + this_bkg_tot
            total_model[i] = this_wh_model

            if this_data_tot >= 50.0:

                # Gaussian limit
                # Under the null hypothesis the data are distributed as a Gaussian with mu = model
                # and sigma = sqrt(model)
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                yerr_low[i] = np.sqrt(this_data_tot)
                yerr_high[i] = np.sqrt(this_data_tot)

            else:

                # Low-counts
                # Under the null hypothesis the data are distributed as a Poisson distribution with
                # mean = model, plot the 68% confidence interval (quantile=[0.16,1-0.16]).
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                quantile = 0.16
                mean = this_wh_model
                y_low = poisson.isf(1-quantile, mu=mean)
                y_high = poisson.isf(quantile, mu=mean)
                yerr_low[i] = mean-y_low
                yerr_high[i] = y_high-mean

        residuals = (total_counts - total_model) / np.sqrt(total_model)
        residuals_err = [yerr_high / np.sqrt(total_model),
                         yerr_low / np.sqrt(total_model)]

        yerr = [yerr_high, yerr_low]

        return self._plot_spectrum(net_counts, yerr, model_only, residuals, residuals_err)

    def _plot_spectrum(self, net_counts, yerr, model_only, residuals, residuals_err):

        fig, subs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})

        subs[0].errorbar(self._active_planes, net_counts, yerr=yerr,
                         capsize=0,
                         color='black', label='Net counts', fmt='.')

        subs[0].plot(self._active_planes, model_only, label='Convolved model')

        subs[0].legend(bbox_to_anchor=(1.0, 1.0), loc="upper right",
                       numpoints=1)

        # Residuals
        subs[1].axhline(0, linestyle='--')

        subs[1].errorbar(
            self._active_planes, residuals,
            yerr=residuals_err,
            capsize=0, fmt='.'
        )

        y_limits = [min(net_counts[net_counts > 0]) / 2., max(net_counts) * 2.]

        subs[0].set_yscale("log", nonposy='clip')
        subs[0].set_ylabel("Counts per bin")
        subs[0].set_xticks([])

        subs[1].set_xlabel("Analysis bin")
        subs[1].set_ylabel(r"$\frac{{cts - mod - bkg}}{\sqrt{mod + bkg}}$")
        subs[1].set_xticks(self._active_planes)
        subs[1].set_xticklabels(self._active_planes)

        subs[0].set_ylim(y_limits)

        return fig

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

        for bin_id in self._active_planes:

            data_analysis_bin = self._maptree[bin_id]

            this_model_map_hpx = self._get_expectation(data_analysis_bin, bin_id, n_point_sources, n_ext_sources)

            # Now compare with observation
            bkg_renorm = self._nuisance_parameters.values()[0].value

            obs = data_analysis_bin.observation_map.as_partial()  # type: np.array
            bkg = data_analysis_bin.background_map.as_partial() * bkg_renorm  # type: np.array

            this_pseudo_log_like = log_likelihood(obs,
                                                  bkg,
                                                  this_model_map_hpx)

            total_log_like += this_pseudo_log_like - self._log_factorials[bin_id] \
                              - self._saturated_model_like_per_maptree[bin_id]

        return total_log_like

    def write(self, response_file_name, map_tree_file_name):
        """
        Write this dataset to disk in HDF format.

        :param response_file_name: filename for the response
        :param map_tree_file_name: filename for the map tree
        :return: None
        """

        self._maptree.write(map_tree_file_name)
        self._response.write(response_file_name)

    def get_simulated_dataset(self, name):
        """
        Return a simulation of this dataset using the current model with current parameters.

        :param name: new name for the new plugin instance
        :return: a HAL instance
        """


        # First get expectation under the current model and store them, if we didn't do it yet

        if self._clone is None:

            n_point_sources = self._likelihood_model.get_number_of_point_sources()
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

            expectations = collections.OrderedDict()

            for bin_id in self._maptree:

                data_analysis_bin = self._maptree[bin_id]
                if bin_id not in self._active_planes:

                    expectations[bin_id] = None

                else:

                    expectations[bin_id] = self._get_expectation(data_analysis_bin, bin_id,
                        n_point_sources, n_ext_sources) + \
                        data_analysis_bin.background_map.as_partial()

            if parallel_client.is_parallel_computation_active():

                # Do not clone, as the parallel environment already makes clones

                clone = self

            else:

                clone = copy.deepcopy(self)

            self._clone = (clone, expectations)

        # Substitute the observation and background for each data analysis bin
        for bin_id in self._clone[0]._maptree:

            data_analysis_bin = self._clone[0]._maptree[bin_id]

            if bin_id not in self._active_planes:

                continue

            else:

                # Active plane. Generate new data
                expectation = self._clone[1][bin_id]
                new_data = np.random.poisson(expectation, size=(1, expectation.shape[0])).flatten()

                # Substitute data
                data_analysis_bin.observation_map.set_new_values(new_data)

        # Now change name and return
        self._clone[0]._name = name
        # Adjust the name of the nuisance parameter
        old_name = self._clone[0]._nuisance_parameters.keys()[0]
        new_name = old_name.replace(self.name, name)
        self._clone[0]._nuisance_parameters[new_name] = self._clone[0]._nuisance_parameters.pop(old_name)

        # Recompute biases
        self._clone[0]._compute_likelihood_biases()

        return self._clone[0]

    def _get_expectation(self, data_analysis_bin, energy_bin_id, n_point_sources, n_ext_sources):

        # Compute the expectation from the model

        this_model_map = None

        for pts_id in range(n_point_sources):

            this_conv_src = self._convolved_point_sources[pts_id]

            expectation_per_transit = this_conv_src.get_source_map(energy_bin_id,
                                                                   tag=None,
                                                                   psf_integration_method=self._psf_integration_method)

            expectation_from_this_source = expectation_per_transit * data_analysis_bin.n_transits

            if this_model_map is None:

                # First addition

                this_model_map = expectation_from_this_source

            else:

                this_model_map += expectation_from_this_source

        # Now process extended sources
        if n_ext_sources > 0:

            this_ext_model_map = None

            for ext_id in range(n_ext_sources):

                this_conv_src = self._convolved_ext_sources[ext_id]

                expectation_per_transit = this_conv_src.get_source_map(energy_bin_id)

                if this_ext_model_map is None:

                    # First addition

                    this_ext_model_map = expectation_per_transit

                else:

                    this_ext_model_map += expectation_per_transit

            # Now convolve with the PSF
            if this_model_map is None:
                
                # Only extended sources
            
                this_model_map = (self._psf_convolutors[energy_bin_id].extended_source_image(this_ext_model_map) *
                                  data_analysis_bin.n_transits)
            
            else:

                this_model_map += (self._psf_convolutors[energy_bin_id].extended_source_image(this_ext_model_map) *
                                   data_analysis_bin.n_transits)


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

    @staticmethod
    def _represent_healpix_map(fig, hpx_map, longitude, latitude, xsize, resolution, smoothing_kernel_sigma):

        proj = get_gnomonic_projection(fig, hpx_map,
                                       rot=(longitude, latitude, 0.0),
                                       xsize=xsize,
                                       reso=resolution)

        if smoothing_kernel_sigma is not None:

            # Get the sigma in pixels
            sigma = smoothing_kernel_sigma * 60 / resolution

            proj = convolve(list(proj),
                            Gaussian2DKernel(sigma),
                            nan_treatment='fill',
                            preserve_nan=True)

        return proj

    def display_fit(self, smoothing_kernel_sigma=0.1, display_colorbar=False):
        """
        Make a figure containing 4 maps for each active analysis bins with respectively model, data,
        background and residuals. The model, data and residual maps are smoothed, the background
        map is not.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel, for all but
        background maps
        :param display_colorbar: whether or not to display the colorbar in the residuals
        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # This is the resolution (i.e., the size of one pixel) of the image
        resolution = 3.0 # arcmin

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        n_active_planes = len(self._active_planes)
        n_columns = 4

        fig, subs = plt.subplots(n_active_planes, n_columns,
                                 figsize=(2.7 * n_columns, n_active_planes * 2))

        with progress_bar(len(self._active_planes), title='Smoothing maps') as prog_bar:

            images = ['None'] * n_columns

            for i, plane_id in enumerate(self._active_planes):

                data_analysis_bin = self._maptree[plane_id]

                # Get the center of the projection for this plane
                this_ra, this_dec = self._roi.ra_dec_center

                this_model_map_hpx = self._get_expectation(data_analysis_bin, plane_id, n_point_sources, n_ext_sources)

                # Make a full healpix map for a second
                whole_map = SparseHealpix(this_model_map_hpx,
                                          self._active_pixels[plane_id],
                                          data_analysis_bin.observation_map.nside).as_dense()

                # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
                longitude = ra_to_longitude(this_ra)

                # Declination is already between -90 and 90
                latitude = this_dec


                # Background and excess maps

                # Make all the projections: model, excess, background, residuals
                proj_model = self._represent_healpix_map(fig, whole_map,
                                                         longitude, latitude,
                                                         xsize, resolution, smoothing_kernel_sigma)
                # Here we removed the background otherwise nothing is visible
                # Get background (which is in a way "part of the model" since the uncertainties are neglected)
                background_map = data_analysis_bin.background_map.as_dense()
                bkg_subtracted = data_analysis_bin.observation_map.as_dense() - background_map
                proj_data = self._represent_healpix_map(fig, bkg_subtracted,
                                                        longitude, latitude,
                                                        xsize, resolution, smoothing_kernel_sigma)
                # No smoothing for this one (because a goal is to check it is smooth).
                proj_bkg = self._represent_healpix_map(fig, background_map,
                                                       longitude, latitude,
                                                       xsize, resolution, None)
                proj_residuals = proj_data - proj_model

                # Common color scale range for model and excess maps
                vmin = min(np.nanmin(proj_model), np.nanmin(proj_data))
                vmax = max(np.nanmax(proj_model), np.nanmax(proj_data))

                # Plot model
                images[0] = subs[i][0].imshow(proj_model, origin='lower', vmin=vmin, vmax=vmax)
                subs[i][0].set_title('model, bin {}'.format(data_analysis_bin.name))

                # Plot data map
                images[1] = subs[i][1].imshow(proj_data, origin='lower', vmin=vmin, vmax=vmax)
                subs[i][1].set_title('excess, bin {}'.format(data_analysis_bin.name))

                # Plot background map.
                images[2] = subs[i][2].imshow(proj_bkg, origin='lower')
                subs[i][2].set_title('background, bin {}'.format(data_analysis_bin.name))

                # Now residuals
                images[3] = subs[i][3].imshow(proj_residuals, origin='lower')
                subs[i][3].set_title('residuals, bin {}'.format(data_analysis_bin.name))

                # Remove numbers from axis
                for j in range(n_columns):
                    subs[i][j].axis('off')

                if display_colorbar:
                    for j, image in enumerate(images):
                        plt.colorbar(image, ax=subs[i][j])

                prog_bar.increase()

        fig.set_tight_layout(True)

        return fig

    def _get_optimal_xsize(self, resolution):

        return 2.2 * self._roi.data_radius.to("deg").value / (resolution / 60.0)

    def display_stacked_image(self, smoothing_kernel_sigma=0.5):
        """
        Display a map with all active analysis bins stacked together.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel to apply
        :return: a matplotlib.Figure instance
        """

        # This is the resolution (i.e., the size of one pixel) of the image in arcmin
        resolution = 3.0

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        active_planes_bins = [self._maptree[x] for x in self._active_planes]

        # Get the center of the projection for this plane
        this_ra, this_dec = self._roi.ra_dec_center

        # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
        longitude = ra_to_longitude(this_ra)

        # Declination is already between -90 and 90
        latitude = this_dec

        total = None

        for i, data_analysis_bin in enumerate(active_planes_bins):

            # Plot data
            background_map = data_analysis_bin.background_map.as_dense()
            this_data = data_analysis_bin.observation_map.as_dense() - background_map
            idx = np.isnan(this_data)
            # this_data[idx] = hp.UNSEEN

            if i == 0:

                total = this_data

            else:

                # Sum only when there is no UNSEEN, so that the UNSEEN pixels will stay UNSEEN
                total[~idx] += this_data[~idx]

        delta_coord = (self._roi.data_radius.to("deg").value * 2.0) / 15.0

        fig, sub = plt.subplots(1, 1)

        proj = self._represent_healpix_map(fig, total, longitude, latitude, xsize, resolution, smoothing_kernel_sigma)

        sub.imshow(proj, origin='lower')
        sub.axis('off')

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
        """
        Return the number of active bins across all active analysis bins

        :return: number of active bins
        """

        n_points = 0

        for bin_id in self._maptree:
            n_points += self._maptree[bin_id].observation_map.as_partial().shape[0]

        return n_points
