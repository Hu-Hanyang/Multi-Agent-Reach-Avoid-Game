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
reach_set = CylinderShape(grid, [2,3], np.array([8.0, 5.0]), 0.8)

# HJ solver setting
tol = 0.001

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Load the 4D TTR value
ttr_value = np.load(f"{current_directory}/hj_values/dubins4d_general_ttr.npy")

# Example usage:
# Assuming you have:
# grid = Grid(min_bounds, max_bounds, 4, resolutions, [3])
# ttr_value = your_4d_array

# Plot x-y contour with v=1.0 and theta=pi/2
fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))

fig, ax = plot_value_contour(grid, ttr_value, [0, 1], {2: 0.1, 3: math.pi/2}, contour_levels=[0.0, 2.5, 5.0, 7.5], linewidth=3.5, linestyle='-', show_contour_values=True, contour_linewidth=2.5, contour_linestyle=':',)

# 清理图形
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 保存纯净图片
plt.savefig(f"{current_directory}/hj_plots/clean_ttr_plot.png", 
           dpi=300, bbox_inches='tight', pad_inches=0)

print("图片尺寸：16x10厘米，无标签和边框")
