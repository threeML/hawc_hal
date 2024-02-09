""" Module to define a class to store convolved sources within a model"""
from .convolved_extended_source import ConvolvedExtendedSource
from .convolved_point_source import ConvolvedPointSource


class ConvolvedSourcesContainer:
    """Container for convolved sources"""

    def __init__(self) -> None:
        """Generate a list to contain convolved sources within model"""
        self._cache: list[ConvolvedPointSource | ConvolvedExtendedSource] = []

    def reset(self) -> None:
        """Reset number of sources within container"""
        self._cache = []

    def __getitem__(
        self, item: ConvolvedPointSource | ConvolvedExtendedSource
    ) -> ConvolvedPointSource | ConvolvedExtendedSource:
        """Retrieve source from container

        :param item: An astromodels source instance
        :return: Source previously stored in container
        """
        return self._cache[item]  # type: ignore

    def append(
        self, convolved_point_source: ConvolvedPointSource | ConvolvedExtendedSource
    ) -> None:
        """Append a convolved source to the container

        :param convolved_point_source: Convolved point source or extended source
        """
        self._cache.append(convolved_point_source)

    @property
    def n_sources_in_cache(self) -> int:
        """Retrieve the number of sources in the container"""
        return len(self._cache)
