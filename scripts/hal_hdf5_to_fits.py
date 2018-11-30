from hawc_hal.maptree import map_tree_factory

import healpy as hp
import numpy as np
import argparse
import os

cwd=os.getcwd()

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--input', metavar='input', type=str,
                    required=True,
                    help='pick an hdf5 file as input')

parser.add_argument('--output', metavar='output', type=str, 
                    default="map",
                    help='pick a file prefix (e.g. given "test" -> test_binX.fits.gz)')


args = parser.parse_args()

filename=os.path.expandvars(args.input)
outfile=os.path.expandvars(args.output)

if not os.path.isfile(filename):
    print("You must enter a valid file!")
    exit(1)


# Export the entire map tree (full sky)
# this throws a warning for partial map trees but its OK
maptree = map_tree_factory(filename, None)

for i, analysis_bin in enumerate(maptree.analysis_bins_labels):
    #properties
    map_bin    = maptree[analysis_bin]
    nside      = map_bin.nside
    npix       = map_bin.npix
    see_pixels = map_bin.observation_map._pixels_ids

    #what we want
    data  = map_bin.observation_map.as_dense()
    bkg   = map_bin.background_map.as_dense()
    zeros = np.zeros(npix)


    outFileName="{0}_bin{1}.fits.gz".format(outfile,analysis_bin)
    #I probably need to add some header info, but yeah
    hp.fitsfunc.write_map(outFileName, (data, bkg, zeros), partial=True, fits_IDL=False)

    print("File Written: {0}".format(outFileName))



