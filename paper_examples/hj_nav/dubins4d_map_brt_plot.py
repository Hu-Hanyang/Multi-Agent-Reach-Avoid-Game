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

# 4. Set the look-back length and time step
lookback_length = 25.0 
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

brt_value = np.load(f"{current_directory}/hj_values/dubins4d_map_brt.npy")
print(f"The shape of the brt_value function is {brt_value.shape}")


# 第一步：画填充（需要至少2个层级）
# 创建图形时直接设置尺寸
fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))

fig, ax = plot_value_contour(grid=grid,
                             value_function=brt_value,
                             plot_dims=[0,1],
                             fixed_values={2: 0.4, 3: math.pi/4.},
                             contour_levels=[-0.000001, 0.0],  # 两个非常接近的值
                            
                             show_contour_labels=False,  # 填充模式
                             fill_color='lightgray',
                             alpha=0.3,
                             ax=ax)


# 第二步：在同一张图上画等高线（只需要一个层级）
plot_value_contour(grid=grid,
                   value_function=brt_value,
                   plot_dims=[0,1],
                   fixed_values={2: 0.4, 3: math.pi/4.},
                   single_contour_value=0.0,  # 这里可以用单个层级
                   vmin=0.0,
                   vmax=0.001,
                   show_contour_labels=True,  # 线条模式
                   show_contour_values=False,
                   contour_colors='black',
                   contour_linewidth=2,
                   contour_linestyle='--',
                   ax=ax)



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
plt.savefig(f"{current_directory}/hj_plots/clean_plot.png", 
           dpi=300, bbox_inches='tight', pad_inches=0)

print("图片尺寸：16x10厘米，无标签和边框")