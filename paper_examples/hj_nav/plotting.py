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

# Solver core
from odp.solver import HJSolver, TTRSolver
import math
from utils import *


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
# Classification variables error thresholds (adjust based on your application)

# BRT related info
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

dyn = "Dubins4D"

rel_grid = Grid(
        minBounds=brt_grid_info[dyn]["minBounds"],
        maxBounds=brt_grid_info[dyn]["maxBounds"],
        dims=brt_grid_info[dyn]["dims"],
        pts_each_dim=brt_grid_info[dyn]["pts_each_dim"],
        periodicDims=brt_grid_info[dyn]["periodicDims"],
    )

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


rel_brt = np.load("paper_examples/hj_nav/data/BRT_Avoid_604c85e3add06c88c04945f85e60718cdea2e8ed46be781015b3fb2f68102456.npy")

fig, ax= plot_value_contour(grid=rel_grid,
                            value_function=rel_brt,
                            plot_dims=[0, 1],
                            fixed_values={2: 0.1, 3: math.pi/2},
                            goal_center=(0., 0.),
                            goal_radius=0.5,
                            vmin=-5.0, 
                            vmax=0.0,
                            # obstacles=[(-1.0, 1.0, -1.0, 1.0)],
                            save_dir=f"{current_directory}/results/"
                            )