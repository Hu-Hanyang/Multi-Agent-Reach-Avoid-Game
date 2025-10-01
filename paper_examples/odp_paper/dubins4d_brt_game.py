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
from odp.compute_trajectory import spa_deriv, find_sign_change, compute_opt_traj
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
lookback_length = 4.5  # the same as 2014Mo
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Load the 4D TTR value
brt_value = np.load(f"{current_directory}/hj_values/dubins4d_brt.npy")

# 1. Initialize the game
T = 20  # total game time
t = 0.05  # time step
agent_init = np.array([-2.0, 0.0, 0.0, -0.785])

agent_current = agent_init
agent_traj = []
current_value = grid.get_values(brt_value[..., 0], agent_current.copy())

print(f"###### The inital value is {current_value}")

# agent_traj, opt_u, opt_d, t = compute_opt_traj(dubins4d,
#                                          grid,
#                                          brt_value,
#                                          tau,
#                                          (-2.0, 0.0, 0.0, -0.785))
# breakpoint()
for step in range(int(T/t)):
    # Log the states
    agent_traj.append(agent_current)
    
    neg2pos, pos2neg = find_sign_change(grid, brt_value, agent_current, tau)
    current_value = grid.get_values(brt_value[..., 0], agent_current.copy())
    if current_value > 0:
        brt_value = brt_value - current_value
    v = brt_value[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    # final_brt = brt_value[..., 0]
    # v = final_brt[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid.get_indices(agent_current.copy()), v, grid)
    agent_control = dubins4d.optCtrl_inPython(agent_current.copy(), spat_deriv_vector)
    
    # Compute the control
    # spat_deriv = spa_deriv_ttr(indices=grid.get_indices(agent_current.copy()),
    #                 values=brt_value,
    #                 grids=grid,
    #                 periodic_dims=[3])
    # agent_control = dubins4d.optCtrl_inPython(agent_current, spat_deriv)

    agent_current = np.array(list(dubins4d.forward(1.0/t, agent_current, agent_control, (0.0, 0.0))))

    if np.linalg.norm(agent_current[:2] - center) <= radius:
        agent_traj.append(agent_current)
        print(f"########## Game is over. #######")
        break


# Plotting
# x_limit = [-3., 3.]
# y_limit = [-1., 4.]
# fig, ax= plot_value_contour(grid=grid,
#                             value_function=brt_value,
#                             plot_dims=[0, 1],
#                             fixed_values={2: 0.1, 3: math.pi/2},
#                             goal_center=(2., 2.),
#                             goal_radius=0.8,
#                             obstacles=[(-1.0, 1.0, -1.0, 1.0)],
#                             )

plot_trajectory_scatter(agent_traj, save_dir=f"{current_directory}/hj_figures/", scatter_density=3,)  # x_limit=x_limit, y_limit=y_limit
        
    


