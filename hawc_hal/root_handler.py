# This module handle the importing of ROOT
# only use in methods that actually need ROOT

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from ROOT import TEntryList

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray

ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)

import root_numpy
