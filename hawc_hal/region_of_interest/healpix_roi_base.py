from builtins import object
_EQUATORIAL = 'equatorial'
_GALACTIC = 'galactic'

_RING = 'RING'
_NESTED = 'NESTED'


class HealpixROIBase(object):

    def active_pixels(self, nside, system=_EQUATORIAL, ordering=_RING):
        """
        Returns the non-zero elements, i.e., the pixels selected according to this Region Of Interest

        :param nside: the NSIDE of the healpix map
        :param system: the system of the Healpix map, either 'equatorial' or 'galactic' (default: equatorial)
        :param ordering: numbering scheme for Healpix. Either RING or NESTED (default: RING)
        :return: an array of pixels IDs (in healpix RING numbering scheme)
        """

        # Let's transform to lower case (so Equatorial will work, as well as EQuaTorial, or whatever)
        system = system.lower()

        assert system == _EQUATORIAL, "%s reference system not supported" % system

        assert ordering in [_RING, _NESTED], "Could not understand ordering %s. Must be %s or %s" % (ordering,
                                                                                                     _RING,
                                                                                                     _NESTED)

        return self._active_pixels(nside, ordering)

    # This is supposed to be overridden by child classes
    def _active_pixels(self, nside, ordering):  # pragma: no cover

        raise NotImplementedError("You need to implement this")

    def display(self):  # pragma: no cover

        raise NotImplementedError("You need to implement this")

    def to_dict(self):  # pragma: no cover

        raise NotImplementedError("You need to implement this")

    def from_dict(self, data):  # pragma: no cover

        raise NotImplementedError("You need to implement this")
