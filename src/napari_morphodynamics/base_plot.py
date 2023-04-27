from napari_matplotlib.base import NapariMPLWidget
from matplotlib.widgets import SpanSelector
import numpy as np

class DataPlotter(NapariMPLWidget):
    """Subclass of napari_matplotlib NapariMPLWidget for data visualization.
    Attributes:
        axes : matplotlib.axes.Axes
        cursor_pos : tuple of current mouse cursor position in the napari viewer
    """
    def __init__(self, napari_viewer, options=None):
        super().__init__(napari_viewer)
        self.axes = self.canvas.figure.subplots()
        self.cursor_pos = np.array([])
        self.axes.tick_params(colors='white')
       

    def clear(self):
        """
        Clear the canvas.
        """
        #self.axes.clear()
        pass