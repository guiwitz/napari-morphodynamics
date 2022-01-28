
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"



from .napari_gui import MorphoWidget
