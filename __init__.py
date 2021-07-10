"""
Initialize the library.
"""
from . import avotools
from . import seismodel
from .pyavo import PyAvoError

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
