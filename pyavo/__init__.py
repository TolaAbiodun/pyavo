"""
Initialize the library.
:license: MIT
"""
from . import avotools
from . import seismodel

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
