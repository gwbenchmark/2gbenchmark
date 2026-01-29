"""2gbenchmark

Code for simulating 2nd generation gravitational-wave benchmark data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aspire")
except PackageNotFoundError:
    __version__ = "unknown"
