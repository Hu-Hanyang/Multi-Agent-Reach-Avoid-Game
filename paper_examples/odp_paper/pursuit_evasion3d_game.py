import imp
import os
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture, DubinsCar
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction
from odp.compute_trajectory import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math


# Define grid
grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([60, 60, 40]), [2])

# # Implicit function for the initial value function
# Initial_value_f = CylinderShape(grid, [2], np.zeros(3), 1)

# # Look-back length and time step of computation
lookback_length = 4.
t_step = 0.1

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# uMode maximizing means avoiding capture, dMode minimizing means capturing
rel_dyn = DubinsCapture(uMode="max", dMode="min", wMax=1.0, dMax=1.0, speed=1.0)
dubins3d = DubinsCar(x=[0,0,0], wMax=1, speed=1, dMax=[0,0,0])

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

rel_finaltime_hj_value = np.load("paper_examples/odp_paper/hj_values/pursuit_evasion3d_FalseAllTimeSteps_4.0.npy")  # for evader
rel_alltime_hj_value = np.load("paper_examples/odp_paper/hj_values/pursuit_evasion3d_TrueAllTimeSteps_4.0.npy")  # for pursuer

# 1. Initialize the game
T = 4.  # total game time
delta_t = 0.1  # time step

pursuer_init = np.array([3.25, 0.0, -3.14])
evader_init = np.array([0.0, 0., 0.])

pursuer_current = pursuer_init
evader_current = evader_init

pursuer_traj = []
evader_traj = []

# Check the initial HJ value
rel_state = compute_relative_state(pursuer_current, evader_current)
init_value = grid.get_values(rel_finaltime_hj_value, rel_state)
print(f"########## The initial HJ value is {init_value}. ##########")

for step in range(int(T/delta_t)):
    # Log the states
    pursuer_traj.append(pursuer_current.copy())
    evader_traj.append(evader_current.copy())
    
    # Compute the controls for the pursuer and the evader
    rel_state = compute_relative_state(pursuer_current, evader_current)
    # spat_deriv = spa_deriv(grid.get_indices(rel_state), rel_finaltime_hj_value[..., np.newaxis], grid, [2])
    
    # Print out the HJ value
    # print(f"********* The current HJ value is {grid.get_values(rel_finaltime_hj_value, rel_state)}. ********")
    
    # pursuer_control = rel_dyn.optDistb_inPython(pursuer_current, spat_deriv)
    pursuer_control = compute_pursuer_control(hjvalue=rel_alltime_hj_value,
                                              rel_dyn=rel_dyn,
                                              grid=grid,
                                              rel_state=rel_state,
                                              tau=tau)
    # evader_control = rel_dyn.optCtrl_inPython(evader_current, spat_deriv)
    evader_control = -dubins3d.wMax
    
    # Update the states with controls
    pursuer_current = tuple_to_arr(dubins3d.forward(1.0/delta_t, pursuer_current, pursuer_control))
    evader_current = tuple_to_arr(dubins3d.forward(1.0/delta_t, evader_current, evader_control))
    
    # Check the status
    if np.linalg.norm(pursuer_current[:2] - evader_current[:2]) <= 1.:
        pursuer_traj.append(pursuer_current.copy())
        evader_traj.append(evader_current.copy())
        print(f"########## Game is over. #######")
        break
    

plot_trajectories_basic(pursuer_traj, evader_traj, f"{current_directory}/hj_figures/")
# plot_trajectories_with_orientation(pursuer_traj, evader_traj)