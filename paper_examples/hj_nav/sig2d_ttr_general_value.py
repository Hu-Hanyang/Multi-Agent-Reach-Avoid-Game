import time
import os
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D, SingleIntegratorConstantSpeed
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction
from odp.compute_trajectory import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, TTRSolver
import math


# # Define grid
grid = Grid(np.array([0.0, 0.0]), np.array([6.0, 8.0]), 2, np.array([160, 100]), [])

# Define my object
my_car = SingleIntegratorConstantSpeed(x=[0,0], uMin=-1, uMax=1, uMode="min", speed=1.0)

# Use the grid to initialize initial value function
reach_set = CylinderShape(grid, [], np.array([2.0, 6.0]), 0.8)
# avoid_set = ShapeRectangle(grid, [0.0, 3.0], [3.0, 5.0])


# HJ solver setting
tol = 0.001

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0,1],
#                   slicesCut=[10, 18], save_fig=True, filename=f"{current_directory}/hj_plots/dubins4d_ttr_map1.png")

# ttr_value = TTRSolver(my_car, grid, [goal_area, obs_area], tol, po)
ttr_value = TTRSolver(my_car, grid, reach_set, tol)


np.save(f"{current_directory}/hj_values/sig2d_ttr_general.npy", ttr_value)
