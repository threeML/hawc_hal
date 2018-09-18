from healpix_cone_roi import HealpixConeROI, HealpixROIBase
from healpix_map_roi import HealpixMapROI


def get_roi_from_dict(dictionary):
    """
    Make a ROI from a dictionary such as the one read from the hd5 file of the maptree.

    :param dictionary:
    :return:
    """

    roi_type = dictionary['ROI type']

    return globals()[roi_type].from_dict(dictionary)
