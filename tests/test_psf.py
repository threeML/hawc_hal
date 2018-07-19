import copy
import pytest
from hawc_hal.psf_fast import InvalidPSF, InvalidPSFError

def test_invalid_psf():

    psf = InvalidPSF()

    # Check that we can copy it
    cpsf = copy.deepcopy(psf)

    # Check that another method raises the desired error
    with pytest.raises(InvalidPSFError):
        print cpsf.repr()
