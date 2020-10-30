from builtins import object
class TF1Wrapper(object):

    def __init__(self, tf1_instance):
        # Make a copy so that if the passed instance was a pointer from a TFile,
        # it will survive the closing of the associated TFile

        self._tf1 = tf1_instance.Clone()

    @property
    def name(self):
        return self._tf1.GetName()

    def integral(self, *args, **kwargs):
        return self._tf1.Integral(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._tf1.Eval(*args, **kwargs)