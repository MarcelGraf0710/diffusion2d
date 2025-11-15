"""
Solving the two-dimensional diffusion equation

Example acquired from https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

### Imports

# NumPy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt

# Time
import time

# Output printing
from .output import create_plot, output_plot 

def do_timestep(u_nm1, u, D, dt, dx2, dy2):
    """
    Propagate with forward-difference in time, central-difference in space
    """
    u[1:-1, 1:-1] = u_nm1[1:-1, 1:-1] + D * dt * (
            (u_nm1[2:, 1:-1] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[:-2, 1:-1]) / dx2
            + (u_nm1[1:-1, 2:] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[1:-1, :-2]) / dy2)

    u_nm1 = u.copy()
    return u_nm1, u


def solve(
        w: float = 10.0, 
        h: float = 10.0, 
        dx: float = 0.1, 
        dy: float = 0.1, 
        D: float = 4.0, 
        T_cold: float = 300.0, 
        T_hot: float = 1200.0,
        print_time_step_width: bool = False,
        print_avg_calc_time: bool = False
    ):
    """Solves the diffusion equation for the specified scenario and shows a plot of the results in a separate window.

    Args:
        w                       (float, optional):  Plate width in mm. Defaults to 10.0.
        h                       (float, optional):  Plate height in mm. Defaults to 10.0.
        dx                      (float, optional):  Interval length in x-direction in mm. Defaults to 0.1.
        dy                      (float, optional):  Interval length in y-direction in mm. Defaults to 0.1.
        D                       (float, optional):  Thermal diffusivity of steel in mm^2/s. Defaults to 4.0.
        T_cold                  (float, optional):  Initial cold temperature of square domain. Defaults to 300.0.
        T_hot                   (float, optional):  Initial hot temperature of circular disc at the center. 
                                                    Defaults to 1200.0.
        print_time_step_width   (bool, optional):   Whether thetime step width should be printed to the console. 
                                                    Defaults to False.
        print_avg_calc_time     (bool, optional):   Whether the average time required to compute a time step should be 
                                                    printed to the console. Defaults to False.
    """

    # Number of discrete mesh points in X and Y directions
    nx, ny = int(w / dx), int(h / dy)

    # Computing a stable time step
    dx2, dy2 = dx * dx, dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    if print_time_step_width:
        print("dt = {}".format(dt))

    u0 = T_cold * np.ones((nx, ny))
    u = u0.copy()

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r = min(h, w) / 4.0
    cx = w / 2.0
    cy = h / 2.0
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u0[i, j] = T_hot

    # Number of timesteps
    nsteps = 101
    # Output 4 figures at these timesteps
    n_output = [0, 10, 50, 100]
    fig_counter = 0
    fig = plt.figure()

    total_time = 0

    # Time loop
    for n in range(nsteps):
        start = time.perf_counter()
        u0, u = do_timestep(u0, u, D, dt, dx2, dy2)
        end = time.perf_counter()
        total_time += end - start

        # Create figure
        if n in n_output:
            fig_counter += 1
            im = create_plot(fig, fig_counter, T_cold, T_hot, n, dt, u)

    # Plot output figures
    output_plot(fig, im)

    total_time /= nsteps

    if print_avg_calc_time:
        print(f"Average time spent performing one time step: {total_time:.6f} seconds")