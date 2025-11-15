"""
Contains functionality for creating subplots showing the solution at different times and for creating an output plot
in a separate window.
"""

# NumPy
import numpy as np
from numpy.typing import NDArray

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def create_plot(fig: Axes, fig_counter: int, T_cold: float, T_hot: float, n: int, dt: float, u: NDArray) -> AxesImage:
    """Adds a subplot to the specified figure.

    Args:
        fig         (Axes):     the figure to which the subplot is added
        fig_counter (int):      the number of the subplot
        T_cold      (float):    lowest temperature for the lower scale of the colormap
        T_hot       (float):    hottest temperature for the upper scale of the colormap
        n           (int):      current time step
        dt          (float):    time step width
        u           (NDArray):  the plot data
    """
    ax = fig.add_subplot(220 + fig_counter)
    im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=T_cold, vmax=T_hot)
    ax.set_axis_off()
    ax.set_title('{:.1f} ms'.format(n * dt * 1000))
    return im


def output_plot(fig: Axes, im: AxesImage) -> None:
    """Shows the figure in a separate window.

    Args:
        fig      (Axes):        the image axes
        im       (AxesImage):   the image data
    """
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()