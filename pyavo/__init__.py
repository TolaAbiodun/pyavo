"""
Initialize the library.
:license: MIT
"""
from pyavo import avotools
from pyavo import seismodel
from .avotools import impedance, gassmann_substitution, log_crossplot, moduli
from .seismodel import angle_stack, tuning_wedge, tuning_prestack, RPTM, wavelet

from .pyavo import PyAvoError
__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
