import time
import os
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction
from odp.compute_trajectory import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, TTRSolver
import math


# # Define grid
grid = Grid(np.array([0.0, 0.0, -0.1, -math.pi]), np.array([16.0, 10.0, 0.8, math.pi]), 4, np.array([160, 100, 50, 36]), [3])

# Define my object
my_car = DubinsCar4D(x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [0.0, 0.0], dMax=[0.0, 0.0], uMode="min", dMode="max")

# Use the grid to initialize initial value function
reach_set = CylinderShape(grid, [2,3], np.array([13.5, 6.5]), 0.8)

obs1 = ShapeRectangle(grid, [0.0, 0.0, -0.1, -math.pi], [16.0, 1.0, 0.8, math.pi])
obs2 = ShapeRectangle(grid, [15.0, 1.0, -0.1, -math.pi], [16.0, 10.0, 0.8, math.pi])
obs12 = np.minimum(obs1, obs2)
del obs1
del obs2
obs3 = ShapeRectangle(grid, [0.0, 4.5, -0.1, -math.pi], [5.0, 5.5, 0.8, math.pi])
obs4 = ShapeRectangle(grid, [7.5, 4.5, -0.1, -math.pi], [9.0, 5.5, 0.8, math.pi])
obs34 = np.minimum(obs3, obs4)
del obs3
del obs4
obs5 = ShapeRectangle(grid, [9.0, 4.5, -0.1, -math.pi], [11.0, 7.0, 0.8, math.pi])
obs6 = ShapeRectangle(grid, [9.0, 8.0, -0.1, -math.pi], [11.0, 10.0, 0.8, math.pi])
obs56 = np.minimum(obs5, obs6)
del obs5
del obs6
avoid_set = np.minimum(np.minimum(obs12, obs34), obs56)
del obs12
del obs34
del obs56

# 4. Set the look-back length and time step
lookback_length = 25.0 
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1], slicesCut=[10])

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"}
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1], slicesCut=[2, 2])

# 5. Call HJSolver function
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
# compMethods = {"TargetSetMode": "minVWithVTarget"}

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

accuracy = "medium"
brt_result = HJSolver(my_car, grid, [reach_set, avoid_set], tau, compMethods, saveAllTimeSteps=False, accuracy=accuracy) # original one

np.save(f"{current_directory}/hj_values/dubins4d_map_brt.npy", brt_result)
