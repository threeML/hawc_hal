import numpy as np
from healpy import projaxes as PA
import matplotlib.pyplot as plt


def get_gnomonic_projection(figure, hpx_map, **kwargs):
    """
    Returns an array containing the Gnomonic projection of the provided Healpix map.

    This is equivalent to hp.gnomview of Healpy BUT the projected array is NOT plotted in the figure, so you can
    plot it later on.

    :param figure: a matplotlib Figure
    :param hpx_map: the healpix map
    :param **kwargs: keywords accepted by hp.gnomview
    :return: the array containing the projection.
    """

    defaults = {'coord': 'C',
                'rot': None,
                'format': '%g',
                'flip': 'astro',
                'xsize': 200,
                'ysize': None,
                'reso': 1.5,
                'nest': False,
                'min': None,
                'max': None,
                'cmap': None,
                'norm': None}

    for key, default_value in list(defaults.items()):

        if key not in kwargs:

            kwargs[key] = default_value

    ## Colas, 2018-07-11: The following fails for really tall figures,
    ## as happens with 2D binning. Top ends up negative, probably matplotlib bug.
    ## So hard code extent instead. Keep the code for now if we want to fix it.
    # left, bottom, right, top = np.array(plt.gca().get_position()).ravel()
    # extent = (left, bottom, right - left, top - bottom)
    # margins = (0.01, 0.0, 0.0, 0.02)
    # extent = (extent[0] + margins[0],
    #           extent[1] + margins[1],
    #           extent[2] - margins[2] - margins[0],
    #           extent[3] - margins[3] - margins[1])
    extent = (0.05, 0.05, 0.9, 0.9)

    ax = PA.HpxGnomonicAxes(figure, extent,
                            coord=kwargs['coord'],
                            rot=kwargs['rot'],
                            format=kwargs['format'],
                            flipconv=kwargs['flip'])

    # Suppress warnings about nans
    with np.warnings.catch_warnings():

        np.warnings.filterwarnings('ignore')

        img = ax.projmap(hpx_map,
                         nest=kwargs['nest'],
                         coord=kwargs['coord'],
                         vmin=kwargs['min'],
                         vmax=kwargs['max'],
                         xsize=kwargs['xsize'],
                         ysize=kwargs['ysize'],
                         reso=kwargs['reso'],
                         cmap=kwargs['cmap'],
                         norm=kwargs['norm'])

    return img
