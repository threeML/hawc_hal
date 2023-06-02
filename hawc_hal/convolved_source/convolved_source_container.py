from builtins import object
class ConvolvedSourcesContainer(object):
    def __init__(self):

        self._cache = []

    def reset(self):

        self._cache = []

    def __getitem__(self, item):

        return self._cache[item]

    def append(self, convolved_point_source):

        self._cache.append(convolved_point_source)

    @property
    def n_sources_in_cache(self):

        return len(self._cache)
