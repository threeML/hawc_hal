import healpy as hp
import numpy as np


def radec_to_vec(ra, dec):

    assert 0 <= ra <= 360
    assert -90 <= dec <= 90

    # Healpix uses the convention -180 - 180 for longitude, instead
    # we get RA between 0 and 360, so we need to wrap
    wrap_angle = 180.0

    lon = np.mod(ra - wrap_angle, 360.0) - (360.0 - wrap_angle)

    vec = hp.dir2vec(lon, dec, lonlat=True)

    return vec


# def pixid_to_radec(nside, pixid, nest=False):
#
#     theta, phi = hp.pix2ang(nside, pixid, nest=nest, lonlat=False)
#
#     ra = np.rad2deg(phi)
#     dec = np.rad2deg(0.5 * np.pi - theta)
#
#     return ra, dec