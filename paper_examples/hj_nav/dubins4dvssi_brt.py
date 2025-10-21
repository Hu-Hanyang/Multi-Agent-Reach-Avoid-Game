import time
import os
from pathlib import Path
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D
# Plot options
from odp.Plots import PlotOptions, visualize_plots
from odp.compute_trajectory import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, TTRSolver
import math

# HJ_Nav repository
from dynamics.Dubins4DvsSI import Dubins4DvsSI
from utils import *
from plot_utils import *

# Grid definition
NUM_SPEED = 50
NUM_THETA = 50
NUM_X = 50
NUM_Y = 50
X_RANGE = [-2.0, 2.0]
Y_RANGE = [-2.0, 2.0]
SPEED_BOUND = [-0.3, 1.2]
HUMAN_SPEED = 0.6
THETA_RANGE = [0.0, 2 * np.pi]
ANGULAR_SPEED_RANGE = [-0.5, 0.5]
# Rectangle info
LENGTH = 0.0  # 0.8 is enough for the approximate 2D rectangle
WIDTH = 0.2

brt_grid_info = {
    "Dubins4D": {
        "minBounds": np.array([X_RANGE[0], Y_RANGE[0], SPEED_BOUND[0], THETA_RANGE[0]]),
        "maxBounds": np.array([X_RANGE[1], Y_RANGE[1], SPEED_BOUND[1], THETA_RANGE[1]]),
        "dims": 4,
        "pts_each_dim": np.array([NUM_X, NUM_Y, NUM_SPEED, NUM_THETA]),
        "periodicDims": [3],
        "speed_vis": 0.8,
        "theta_vis": 3 * np.pi / 2.0,  # for visualization
    },
    "Dubins3D": {
        "minBounds": np.array([X_RANGE[0], Y_RANGE[0], THETA_RANGE[0]]),
        "maxBounds": np.array([X_RANGE[1], Y_RANGE[1], THETA_RANGE[1]]),
        "dims": 3,
        "pts_each_dim": np.array([NUM_X, NUM_Y, NUM_THETA]),
        "periodicDims": [2],
        "theta_vis": 3 * np.pi / 2.0,  # for visualization
    },
}


# Necessary conditions
dyn = "Dubins4D"
capture_radius = 0.0
horizon = 2.0
v_plot = 0.5
theta_plot = np.pi/2.
v_slice = value_to_slice(v_plot, SPEED_BOUND, NUM_SPEED)
theta_slice = value_to_slice(theta_plot, THETA_RANGE, NUM_THETA)
save_all_time = False
threshold = 0.35
if capture_radius == 0.0:
    threshold = 0.35
else:
    threshold = capture_radius

# Define Grid and Dynamics
rel_dyn = Dubins4DvsSI(x=np.zeros(4),
                       uMin=[-1, -1],
                       uMax=[1., 1.],
                       dMin=np.array([-1.0, -0.5]),
                       dMax=np.array([1.0, 0.5]),
                       uMode="min",  # pursuer tries to minimize
                       dMode="max"  # evader (robot) tries to maximize
                       )

grid = Grid(minBounds=brt_grid_info[dyn]["minBounds"],
            maxBounds=brt_grid_info[dyn]["maxBounds"],
            dims=brt_grid_info[dyn]["dims"],
            pts_each_dim=brt_grid_info[dyn]["pts_each_dim"],
            periodicDims=brt_grid_info[dyn]["periodicDims"]
            )

# Define target and avoid sets
target_set = []
avoid_set = CylinderShape(grid=grid, 
                          center=np.array([0.0, 0.0]),
                          radius=capture_radius,
                          ignore_dims=[2, 3],
                          quadratic=True)  # sqare of relative distance


# Compute the HJ value function
lookback_length = horizon
t_step = 0.025
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

po = PlotOptions(do_plot=True, 
                 plot_type="set",
                 plotDims=[0, 1],
                 slicesCut=[v_slice, theta_slice],
                 save_fig=True,
                 filename=f"{current_directory}/hj_values/dubins4dvssi_horizon{horizon}_radius{capture_radius}_v{v_plot}_theta{theta_plot:.2f}_saveAllTime{save_all_time}",
                 interactive_html=True
                 )

compMethods = {"TargetSetMode": "minVWithV0"}  # BRT
accuracy = "medium"

hj_value_name = f"{current_directory}/hj_values/dubins4dvssi_horizon{horizon}_radius{capture_radius}_v{v_plot}_theta{theta_plot:.02f}_saveAllTime{save_all_time}.npy"

# Check whether we have this file or not
if os.path.exists(hj_value_name):
    hj_value = np.load(hj_value_name)
else:
    hj_value = HJSolver(dynamics_obj=rel_dyn,
                        grid=grid,
                        multiple_value=avoid_set,
                        tau=tau,
                        compMethod=compMethods,
                        saveAllTimeSteps=save_all_time,
                        accuracy=accuracy)
    np.save(hj_value_name, hj_value)
    print(f" ########## The HJ value function is saved as {hj_value_name}. ##########")


if not save_all_time:

    human_position = np.array([5.0, 7.0])
    robot_state = np.array([5.0, 5.0, v_plot, theta_plot])

    fig, ax = visualize_human_robot(grid=grid,
                                    value_function=hj_value,
                                    plot_dims=[0, 1,],
                                    fixed_values={2: v_plot, 3: theta_plot},
                                    human_position=human_position,
                                    robot_state=robot_state,
                                    threshold=threshold,
                                    fig=None,
                                    ax=None)

    fig_name = f"{current_directory}/hj_plots/dubins4dvssi_horizon{horizon}_radius{capture_radius}_threshold{threshold}_v{v_plot}_theta{theta_plot:.02f}_saveAllTime{save_all_time}.png"
    fig.savefig(fig_name)
    print(f"********** The figure is saved as: {fig_name}. *********")

visualize_plots(hj_value, grid, po)
