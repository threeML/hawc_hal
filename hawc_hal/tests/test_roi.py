import pytest
from hawc_hal import HealpixConeROI
from hawc_hal import HealpixMapROI
from hawc_hal.region_of_interest import get_roi_from_dict
import healpy as hp
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import os


def Sky2Vec(ra, dec):
    c = SkyCoord(frame="icrs", ra=ra * u.degree, dec=dec * u.degree)
    theta = (90.0 * u.degree - c.dec).to(u.radian).value
    phi = c.ra.to(u.radian).value
    vec = hp.pixelfunc.ang2vec(theta, phi)
    return vec


NSIDE = 512


def test_rois():

    # test to make sure ConeROI and MapROI with a simple, circular active region return the same active pixels.

    ra, dec = 100, 30
    data_radius = 10
    model_radius = 15

    cone_roi = HealpixConeROI(
        data_radius=data_radius, model_radius=model_radius, ra=ra, dec=dec
    )

    m = np.zeros(hp.nside2npix(NSIDE))
    vec = Sky2Vec(ra, dec)
    m[
        hp.query_disc(
            NSIDE, vec, (data_radius * u.degree).to(u.radian).value, inclusive=False
        )
    ] = 1

    hp.fitsfunc.write_map(
        "roitemp.fits", m, nest=False, coord="C", partial=False, overwrite=True
    )

    map_roi = HealpixMapROI(
        data_radius=data_radius, ra=ra, dec=dec, model_radius=model_radius, roimap=m
    )
    fits_roi = HealpixMapROI(
        data_radius=data_radius,
        ra=ra,
        dec=dec,
        model_radius=model_radius,
        roifile="roitemp.fits",
    )

    assert np.all(cone_roi.active_pixels(NSIDE) == map_roi.active_pixels(NSIDE))
    assert np.all(fits_roi.active_pixels(NSIDE) == map_roi.active_pixels(NSIDE))

    os.remove("roitemp.fits")
    # test that all is still good after saving the ROIs to dictionaries and restoring.
    cone_dict = cone_roi.to_dict()
    map_dict = map_roi.to_dict()
    fits_dict = fits_roi.to_dict()

    cone_roi2 = get_roi_from_dict(cone_dict)
    map_roi2 = get_roi_from_dict(map_dict)
    fits_roi2 = get_roi_from_dict(fits_dict)

    assert np.all(cone_roi2.active_pixels(NSIDE) == map_roi.active_pixels(NSIDE))
    assert np.all(fits_roi2.active_pixels(NSIDE) == map_roi.active_pixels(NSIDE))
    assert np.all(map_roi2.active_pixels(NSIDE) == map_roi.active_pixels(NSIDE))
