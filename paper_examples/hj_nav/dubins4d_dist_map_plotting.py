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

def create_4d_distance_heuristic_vectorized(grid, goal_center, goal_radius):
    """
    向量化版本 - 更高效
    """
    # 获取网格信息
    min_bounds = grid.min
    max_bounds = grid.max
    resolutions = grid.pts_each_dim
    
    # 生成坐标数组
    coords = [np.linspace(min_bounds[i], max_bounds[i], resolutions[i]) for i in range(4)]
    x_coords, y_coords, v_coords, theta_coords = coords
    
    # 创建网格
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # 计算距离（向量化操作）
    distances = np.sqrt((X - goal_center[0])**2 + (Y - goal_center[1])**2) - goal_radius
    
    # 将圆内的值设为0（距离为负值表示在圆内）
    distances = np.maximum(distances, 0.0)
    
    # 计算启发式值
    heuristic_2d = distances / 1.0
    
    # 使用广播机制扩展到4D
    value_4d = heuristic_2d[:, :, np.newaxis, np.newaxis] * np.ones((1, 1, len(v_coords), len(theta_coords)))
    
    return value_4d

goal_center = np.array([2.0, 6.0])
goal_radius = 0.8

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
# accurate_map_ttr = np.load(f"{current_directory}/hj_values/dubins4d_ttr_map1.npy")
# general_ttr = np.load(f"{current_directory}/hj_values/dubins4d_ttr_map2.npy")
# ttrs = [accurate_map_ttr, general_ttr]
# ttr_names = ["accurate_map_ttr", "general_ttr"]  # 添加对应的名称列表

distance_heuristic_4d = create_4d_distance_heuristic_vectorized(grid, goal_center, goal_radius)

# # 创建字典来映射ttr到名称
# ttr_dict = {id(ttr): name for ttr, name in zip(ttrs, ttr_names)}

# for ttr in ttrs:
    # Plot x-y contour with v=1.0 and theta=pi/2
fig, ax = plt.subplots(figsize=(6/2.54, 8/2.54))

contour_levels = np.arange(0.0, 10.0, 0.5)

fig, ax = plot_value_contour(grid, distance_heuristic_4d, [0, 1], {2: 0.4, 3: math.pi/2}, linewidth=3.5, linestyle='-', show_contour_values=True, contour_levels=contour_levels, contour_linewidth=2.5, contour_linestyle=':', contour_label_fontsize=15)

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

# ttr_name = ttr_dict[id(ttr)]

# 保存纯净图片
plt.savefig(f"{current_directory}/hj_plots/distance_contour.png", 
        dpi=300, bbox_inches='tight', pad_inches=0)

print(f"The figure is saved at: {current_directory}/hj_plots/distance_contour.png")
