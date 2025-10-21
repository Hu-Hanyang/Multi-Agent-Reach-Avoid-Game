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

from typing import Optional, List, Dict, Union


# Grid definition
NUM_SPEED = 50
NUM_THETA = 50
NUM_X = 50
NUM_Y = 50
X_RANGE = [-5.0, 5.0]
Y_RANGE = [-5.0, 5.0]
# SPEED_BOUND = [-0.3, 1.2]
HUMAN_SPEED = 1.0
SPEED_BOUND = [-0.3, 1.2]
THETA_RANGE = [0.0, 2 * np.pi]
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

def add_speed_limits_Dubins4DvsSI(
    grid: Grid,
    v_range: np.ndarray,
    avoid_set_shapes,
):
    """
    Adds shapes representing the target set and avoid set to account for speed limits
    at the goal and in general for the Dubins4D system

    Args:
        grid (Grid): Grid on which the shapes are defined
        v_range (np.ndarray): speed range (lower and upper velocity limits)
        avoid_set_shapes (List[Shape]): list of existing shapes in avoid set

    Returns:
        tuple[List[Shape], List[Shape]]:
            Tuple containing updated target_set_shapes and avoid_set_shapes
    """
    # ===================
    # Overall speed limit
    # ===================
    lower_speed_limit_shape = ShapeRectangle(
        grid=grid,
        target_min=np.array(
            [
                grid.min[0],
                grid.min[1],
                SPEED_BOUND[0],
                grid.min[3],
            ]
        ),
        target_max=np.array(
            [
                grid.max[0],
                grid.max[1],
                v_range[0] - 0.05,
                grid.max[3],
            ]
        ),
    )
    avoid_set_shapes.append(lower_speed_limit_shape)

    upper_speed_limit_shape = ShapeRectangle(
        grid=grid,
        target_min=np.array(
            [
                grid.min[0],
                grid.min[1],
                v_range[1] + 0.05,
                grid.min[3],
            ]
        ),
        target_max=np.array(
            [
                grid.max[0],
                grid.max[1],
                SPEED_BOUND[1],
                grid.max[3],
            ]
        ),
    )
    avoid_set_shapes.append(upper_speed_limit_shape)

    return avoid_set_shapes


# Necessary conditions
dyn = "Dubins4D"
v_range = np.array([-0.2, 0.8])
# Pursuit-Evason game setting
capture_radius = 0.0
horizon = 2.0
threshold = 0.35
if capture_radius == 0.0:
    threshold = 0.6
else:
    threshold = capture_radius
# Robot fixed states
v_plot = 0.6
theta_plot = 0.0
v_slice = value_to_slice(v_plot, SPEED_BOUND, NUM_SPEED)
theta_slice = value_to_slice(theta_plot, THETA_RANGE, NUM_THETA)
# HJSolver setting
save_all_time = False

# Define Grid and Dynamics
rel_dyn = Dubins4DvsSI(x=np.zeros(4),
                       uMin=[-1, -1],
                       uMax=[1., 1.],
                       dMin=np.array([-1.0, -1.5]),
                       dMax=np.array([1.0, 1.5]),
                       uMode="min",  # pursuer tries to minimize
                       dMode="max",  # evader (robot) tries to maximize
                       speed_SI=HUMAN_SPEED,
                       )

grid = Grid(minBounds=brt_grid_info[dyn]["minBounds"],
            maxBounds=brt_grid_info[dyn]["maxBounds"],
            dims=brt_grid_info[dyn]["dims"],
            pts_each_dim=brt_grid_info[dyn]["pts_each_dim"],
            periodicDims=brt_grid_info[dyn]["periodicDims"]
            )

# Define target and avoid sets
avoid_set_list = []
pursuit_evasion_set = CylinderShape(grid=grid, 
                          center=np.array([0.0, 0.0]),
                          radius=capture_radius,
                          ignore_dims=[2, 3],
                          quadratic=True)  # sqare of relative distance
avoid_set_list.append(pursuit_evasion_set)

avoid_set_list = add_speed_limits_Dubins4DvsSI(grid=grid,
                                          v_range=v_range,
                                          avoid_set_shapes=avoid_set_list)


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

hj_value_name = f"{current_directory}/hj_values/dubins4dvssi_horizon{horizon}_radius{capture_radius}_saveAllTime{save_all_time}.npy"

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

# visualize_plots(hj_value, grid, po)
