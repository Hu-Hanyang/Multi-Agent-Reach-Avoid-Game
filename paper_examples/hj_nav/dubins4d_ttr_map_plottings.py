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
grid = Grid(np.array([0.0, 0.0, -0.1, -math.pi]), np.array([6.0, 8.0, 0.8, math.pi]), 4, np.array([160, 100, 50, 36]), [3])

# Define my object
my_car = DubinsCar4D(x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [0.0, 0.0], dMax=[0.0, 0.0], uMode="min", dMode="max")

# Use the grid to initialize initial value function
reach_set = CylinderShape(grid, [2,3], np.array([2.0, 6.0]), 0.8)
avoid_set = ShapeRectangle(grid, [0.0, 3.0, -0.1, -math.pi], [3.0, 5.0, 0.8, math.pi])


# HJ solver setting
tol = 0.001

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Load the 4D TTR value
accurate_map_ttr = np.load(f"{current_directory}/hj_values/dubins4d_ttr_map1.npy")
general_ttr = np.load(f"{current_directory}/hj_values/dubins4d_ttr_map2.npy")
ttrs = [accurate_map_ttr, general_ttr]
ttr_names = ["accurate_map_ttr", "general_ttr"]  # 添加对应的名称列表

contour_levels = np.arange(0.0, 20.0, 0.5)
fixed_values = {2: 0.2, 3: math.pi/2}
sampling_state1 = np.array([2.0, 1.0, 0.2, math.pi/2])
sampling_state2 = np.array([2.0, 1.0, 0.2, -math.pi/2])
sampling_states = np.vstack([sampling_state1, sampling_state2])
print(f"###### The estimated TTR heuristics at sampling states of the accurate_map_ttr is {grid.get_values(accurate_map_ttr, sampling_states)} ######")
print(f"###### The estimated TTR heuristics at sampling states of the general_ttr is {grid.get_values(general_ttr, sampling_states)} ######")
# breakpoint()
# 格式化文件名，将浮点数转换为更友好的格式
# 从fixed_values中提取v和theta值
v_value = fixed_values[2]
theta_value = fixed_values[3]
v_str = f"{v_value:.1f}".replace('.', 'p')  # 0.4 -> "0p4"
theta_str = f"{theta_value:.2f}".replace('.', 'p')  # 1.57 -> "1p57"
# 创建字典来映射ttr到名称
ttr_dict = {id(ttr): name for ttr, name in zip(ttrs, ttr_names)}

for ttr in ttrs:
    # Plot x-y contour with v=1.0 and theta=pi/2
    fig, ax = plt.subplots(figsize=(6/2.54, 8/2.54))

    fig, ax = plot_value_contour(grid, ttr, [0, 1], fixed_values=fixed_values, linewidth=3.5, linestyle='-', show_contour_values=True, contour_levels=contour_levels,  contour_linewidth=2.5, contour_linestyle=':', contour_label_fontsize=15)

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
    
    ttr_name = ttr_dict[id(ttr)]

    # 保存纯净图片
    plt.savefig(f"{current_directory}/hj_plots/{ttr_name}_v{v_str}_theta{theta_str}.png", 
            dpi=300, bbox_inches='tight', pad_inches=0)

    print(f"The figure is saved at: {current_directory}/hj_plots/{ttr_name}_v{v_str}_theta{theta_str}.png")
