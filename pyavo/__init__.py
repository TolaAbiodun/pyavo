"""
Initialize the library.
"""
from pyavo import avotools
from pyavo import seismodel
from .avotools import impedance, gassmann, log_crossplot, moduli
from .seismodel import angle_stack, tuning_wedge, tuning_prestack, rpm, wavelet

from .pyavo import PyAvoError
__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
