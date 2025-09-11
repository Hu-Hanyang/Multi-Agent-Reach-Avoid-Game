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
# from odp.compute_trajectory_TTR import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, TTRSolver
import math


# # Define grid
grid = Grid(np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])

# Define my object
dubins4d = DubinsCar4D(x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [0.0, 0.0], dMax=[0.0, 0.0], uMode="min", dMode="max")

# Use the grid to initialize initial value function
center = np.array([2., 2.])
radius = 0.8
goal_area = CylinderShape(grid, [2,3], center, radius)
obs_area = ShapeRectangle(grid, np.array([-1, -1, -1.0, -10]), np.array([1, 1, 5.0, 10]))

# HJ solver setting
tol = 0.001

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Load the 4D TTR value
ttr_value = np.load(f"{current_directory}/hj_values/dubins4d_ttr.npy")

# 1. Initialize the game
T = 20  # total game time
t = 0.05  # time step
agent_init = np.array([-2.0, 0.0, 0.0, -0.785])

agent_current = agent_init
agent_traj = []

for step in range(int(T/t)):
    # Log the states
    agent_traj.append(agent_current)
    
    # Compute the control
    spat_deriv = spa_deriv_ttr(indices=grid.get_indices(agent_current.copy()),
                    values=ttr_value,
                    grids=grid,
                    periodic_dims=[3])
    agent_control = dubins4d.optCtrl_inPython(agent_current, spat_deriv)

    agent_current = np.array(list(dubins4d.forward(1.0/t, agent_current, agent_control, (0.0, 0.0))))

    if np.linalg.norm(agent_current[:2] - center) <= radius:
        agent_traj.append(agent_current)
        print(f"########## Game is over. #######")
        break

plt = plot_value_contour(grid, ttr_value, [0, 1], {2: 0.1, 3: math.pi/2})

plot_trajectory_basic(agent_traj, plt=plt, save_dir=f"{current_directory}/hj_figures/")
        
    


