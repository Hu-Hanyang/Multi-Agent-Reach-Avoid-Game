import numpy as np
import matplotlib.pyplot as plt
from odp.compute_trajectory import spa_deriv, find_sign_change


def compute_relative_state(pursuer_state: np.ndarray, evader_state: np.ndarray):
    e_theta =  - evader_state[2]
    rotation_matrix = np.array([[np.cos(e_theta), -np.sin(e_theta)], [np.sin(e_theta), np.cos(e_theta)]])
    diff_xy = pursuer_state[:2] - evader_state[:2]
    rel_xy = np.matmul(rotation_matrix, diff_xy)
    rel_state = np.concatenate([rel_xy, np.array([pursuer_state[2]-evader_state[2]])])
    
    return rel_state

#TODO: not verified yet.9.10
def compute_pursuer_control(hjvalue, rel_dyn, grid, rel_state, tau):
    """
    hjvalue (np.array): the hj value function with all time slices
    """
    neg2pos, pos2neg = find_sign_change(grid, hjvalue, rel_state, tau)
    if len(neg2pos):
        current_value = grid.get_value(hjvalue[..., 0], rel_state)
        if current_value > 0:
            hjvalue = hjvalue - current_value
        value = hjvalue[..., neg2pos]
        spat_deriv = spa_deriv(grid.get_index(rel_state), value, grid, [2])
        control = rel_dyn.optDistb_inPython(rel_state, spat_deriv)
    else:
        control = (0.0)
    
    return control
    

def compute_evader_control(hjvalue, rel_dyn, grid, rel_state, tau):
    """
    hjvalue (np.array): the hj value function with all time slices
    """
    neg2pos, pos2neg = find_sign_change(grid, hjvalue, rel_state, tau)
    if len(neg2pos):
        current_value = grid.get_value(hjvalue[..., 0], rel_state)
        if current_value > 0:
            hjvalue = hjvalue - current_value
        value = hjvalue[..., neg2pos]
        spat_deriv = spa_deriv(grid.get_index(rel_state), value, grid, [2])
        control = rel_dyn.optCtrl_inPython(rel_state, spat_deriv)
    else:
        control = (0.0)
    
    return control

def tuple_to_arr(new_state):
    x, y, theta = new_state
    
    return np.array([x, y, theta])


def spa_deriv_ttr(indices, values, grids, periodic_dims=[]):
    """Calculates the spatial derivatives of the values at an index for each dimension

    Args:
        indices (tuple): The indices of the state in the grid dimension.
        values (ndarray): The value function shapes like [..., neg2pos] where neg2pos is a list [scalar] or [].
        grids (class): The instance of the corresponding Grid.
        periodic_dims (list): The corrsponding periodical dimensions [].

    Returns:
        List of left and right spatial derivatives for each dimension.
    """
    spa_derivatives = []
    for dim, idx in enumerate(indices):
        if dim == 0:
            left_index = []
        else:
            left_index = list(indices[:dim])

        if dim == len(indices) - 1:
            right_index = []
        else:
            right_index = list(indices[dim + 1:])

        next_index = tuple(
            left_index + [indices[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [indices[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [values.shape[dim] - 1] + right_index
                )
                left_boundary = values[left_periodic_boundary_index]
            else:
                left_boundary = values[indices] + np.abs(values[next_index] - values[indices]) * np.sign(values[indices])
            left_deriv = (values[indices] - left_boundary) / grids.dx[dim]
            right_deriv = (values[next_index] - values[indices]) / grids.dx[dim]
        elif idx == values.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = values[right_periodic_boundary_index]
            else:
                right_boundary = values[indices] + np.abs(values[indices] - values[prev_index]) * np.sign([values[indices]])
            left_deriv = (values[indices] - values[prev_index]) / grids.dx[dim]
            right_deriv = (right_boundary - values[indices]) / grids.dx[dim]
        else:
            left_deriv = (values[indices] - values[prev_index]) / grids.dx[dim]
            right_deriv = (values[next_index] - values[indices]) / grids.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2))
    return spa_derivatives


def plot_trajectories_with_orientation(p_traj, e_traj, arrow_step=5):
    """
    Plot the heading angle with arrow
    """
    p_array = np.array(p_traj)
    e_array = np.array(e_traj)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制轨迹线
    ax.plot(p_array[:, 0], p_array[:, 1], 'r-', linewidth=2, label='p trajectory', alpha=0.7)
    ax.plot(e_array[:, 0], e_array[:, 1], 'b-', linewidth=2, label='e trajectory', alpha=0.7)
    
    # 绘制方向箭头（每隔 arrow_step 个点画一个）
    arrow_length = 0.5
    for i in range(0, len(p_array), arrow_step):
        if i < len(p_array):
            ax.arrow(p_array[i, 0], p_array[i, 1], 
                    arrow_length * np.cos(p_array[i, 2]), 
                    arrow_length * np.sin(p_array[i, 2]),
                    head_width=0.1, fc='red', ec='red', alpha=0.6)
    
    for i in range(0, len(e_array), arrow_step):
        if i < len(e_array):
            ax.arrow(e_array[i, 0], e_array[i, 1], 
                    arrow_length * np.cos(e_array[i, 2]), 
                    arrow_length * np.sin(e_array[i, 2]),
                    head_width=0.1, fc='blue', ec='blue', alpha=0.6)
    
    # 起点和终点
    ax.plot(p_array[0, 0], p_array[0, 1], 'go', markersize=10, label='p start')
    ax.plot(p_array[-1, 0], p_array[-1, 1], 'rs', markersize=10, label='p end')
    ax.plot(e_array[0, 0], e_array[0, 1], 'go', markersize=10, markerfacecolor='none', label='e start')
    ax.plot(e_array[-1, 0], e_array[-1, 1], 'bs', markersize=10, label='e end')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectories with Orientation')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()


def plot_trajectory_basic(traj, plt=None, save_dir=None):
    """
    Plot one trajectory only
    """
    traj_array = np.array(traj)
    
    if plt is None:
        plt.figure(figsize=(10, 8))
    
    # Plot the trajectory
    plt.plot(traj_array[:, 0], traj_array[:, 1], 'r-', linewidth=2, label='p trajectory')
    
    # Plot initial and goal position
    plt.plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=8, label='p start')
    plt.plot(traj_array[-1, 0], traj_array[-1, 1], 'rs', markersize=8, label='p end')
    
    # Plot the arrow of the initial position
    start_x, start_y = traj_array[0, 0], traj_array[0, 1]
    start_theta = traj_array[0, 3] 
    # the length of the arrow
    arrow_length = 0.2
    dx = arrow_length * np.cos(start_theta)
    dy = arrow_length * np.sin(start_theta)
    plt.arrow(start_x, start_y, dx, dy, 
              head_width=0.1, head_length=0.15, 
              fc='blue', ec='blue', 
              label='Start Heading')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    if save_dir is not None:
        plt.savefig(f"{save_dir}/traj.png")
    plt.show()


def plot_trajectory_scatter(traj, ax=None, save_dir=None, scatter_density=10, x_limit=None, y_limit=None):
    """
    Scatter plot trajectory: Display position and direction using scatter points and arrows
    
    Parameters:
    - traj: Trajectory data containing [x, y, v, theta, ...]
    - ax: matplotlib axes object, if None then create a new figure
    - save_dir: Save directory path, if None then don't save
    - scatter_density: Scatter density, number of points to skip between each scatter point and arrow
    - x_limit: X-axis range [min, max]
    - y_limit: Y-axis range [min, max]
    """
    # Convert to numpy array for indexing
    traj_array = np.array(traj)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Sample every scatter_density points
    sampled_indices = range(0, len(traj_array), scatter_density)
    sampled_traj = traj_array[sampled_indices]
    
    # Plot scatter points
    ax.scatter(sampled_traj[:, 0], sampled_traj[:, 1], 
               c='red', marker='o', s=30, alpha=0.7, label='Position points')
    
    # Plot start and end points (special markers)
    ax.scatter(traj_array[0, 0], traj_array[0, 1], 
               c='green', marker='o', s=100, label='Start')
    ax.scatter(traj_array[-1, 0], traj_array[-1, 1], 
               c='blue', marker='s', s=100, label='End')
    
    # Draw direction arrows for sampled points
    arrow_length = 0.3  # Arrow length
    
    for i, point in enumerate(sampled_traj):
        x, y, theta = point[0], point[1], point[3]
        
        # Calculate arrow direction vector
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        
        # Draw arrow
        ax.arrow(x, y, dx, dy, 
                 head_width=0.08, head_length=0.12, 
                 fc='purple', ec='purple', alpha=0.6,
                 length_includes_head=True)
    
    # # Special annotation for start direction (larger and more prominent arrow)
    # start_x, start_y = traj_array[0, 0], traj_array[0, 1]
    # start_theta = traj_array[0, 3]
    # start_dx = arrow_length * 1.5 * np.cos(start_theta)
    # start_dy = arrow_length * 1.5 * np.sin(start_theta)
    
    # ax.arrow(start_x, start_y, start_dx, start_dy, 
    #          head_width=0.15, head_length=0.2, 
    #          fc='darkblue', ec='darkblue', 
    #          label='Start Heading')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('DubinsCar4D with TTR-generated trajectory')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Set axis limits
    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    
    if save_dir is not None:
        fig.savefig(f"{save_dir}/traj_scatter.png", dpi=300, bbox_inches='tight')
        print(f"********** The figure is saved as:{save_dir}/traj_scatter.png. ***********")
    
    return fig, ax


def plot_trajectories_basic(p_traj, e_traj, save_dir=None):
    """Plot 2 trajectories

    Args:
        p_traj (_type_): _description_
        e_traj (_type_): _description_
        save_dir (_type_, optional): _description_. Defaults to None.
    """
    
    p_array = np.array(p_traj)
    e_array = np.array(e_traj)
    
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    plt.plot(p_array[:, 0], p_array[:, 1], 'r-', linewidth=2, label='p trajectory')
    plt.plot(e_array[:, 0], e_array[:, 1], 'b-', linewidth=2, label='e trajectory')
    
    # Plot initial and goal positions
    plt.plot(p_array[0, 0], p_array[0, 1], 'go', markersize=8, label='p start')
    plt.plot(p_array[-1, 0], p_array[-1, 1], 'rs', markersize=8, label='p end')
    plt.plot(e_array[0, 0], e_array[0, 1], 'go', markersize=8, markerfacecolor='none', label='e start')
    plt.plot(e_array[-1, 0], e_array[-1, 1], 'bs', markersize=8, label='e end')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    if save_dir is not None:
        plt.savefig(f"{save_dir}/trajs.png")
    plt.show()


def plot_value_contour(grid, value_function, plot_dims, fixed_values=None, vmin=0, vmax=10, 
                      ax=None, save_dir=None, goal_center=None, goal_radius=None, 
                      obstacles=None, obstacle_color='red', goal_color='green',
                      linewidth=2, linestyle='-', contour_linewidth=2, contour_linestyle='-',
                      show_contour_labels=True, contour_levels=20):
    """
    Plot contour lines of a 4D value function with goal and obstacles (boundaries only).
    
    Parameters:
    -----------
    grid : Grid object
        Grid object with min, max, and pts_each_dim attributes
    value_function : numpy array
        4D array of shape (x_res, y_res, v_res, theta_res)
    plot_dims : list of int
        Dimensions to plot on x and y axes [x_dim, y_dim]
    fixed_values : dict, optional
        Dictionary specifying fixed values for non-plotted dimensions
        Example: {2: 1.0, 3: math.pi/2} for v=1.0, theta=pi/2
    ax : matplotlib axes object, optional
        Axes to plot on. If None, creates new figure.
    vmin, vmax : float
        Value range for contour levels
    goal_center : tuple, optional
        (x, y) coordinates of the goal center
    goal_radius : float, optional
        Radius of the goal circle
    obstacles : list of tuples, optional
        List of obstacle rectangles as [(x_min, x_max, y_min, y_max), ...]
    obstacle_color : str, optional
        Color for obstacle boundaries
    goal_color : str, optional
        Color for goal boundary
    linewidth : float, optional
        Line width for goal and obstacle boundaries
    linestyle : str, optional
        Line style for goal and obstacle boundaries
    contour_linewidth : float, optional
        Line width for contour lines
    contour_linestyle : str, optional
        Line style for contour lines
    show_contour_labels : bool, optional
        Whether to show contour value labels
    contour_levels : int or array-like, optional
        Number of contour levels or specific level values
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Default fixed values if not provided
    if fixed_values is None:
        fixed_values = {2: 1.0, 3: np.math.pi/2}  # Default: v=1.0, theta=pi/2
    
    # Get grid information
    min_bounds = grid.min
    max_bounds = grid.max
    resolutions = grid.pts_each_dim
    
    # Create coordinate arrays for each dimension
    coords = []
    for i in range(4):
        coords.append(np.linspace(min_bounds[i], max_bounds[i], resolutions[i]))
    
    # Find indices for fixed values
    fixed_indices = {}
    for dim, value in fixed_values.items():
        fixed_indices[dim] = np.argmin(np.abs(coords[dim] - value))
    
    # Extract the 2D slice
    slice_indices = [slice(None)] * 4  # Start with full slices for all dimensions
    
    # Set fixed dimensions to their specific indices
    for dim, idx in fixed_indices.items():
        slice_indices[dim] = idx
    
    # Convert to tuple for indexing
    slice_indices = tuple(slice_indices)
    slice_2d = value_function[slice_indices]
    
    # Clip the values to the specified range
    slice_2d_clipped = np.clip(slice_2d, vmin, vmax)
    
    # Transpose if necessary to get correct orientation
    if plot_dims != [0, 1]:
        # We need to rearrange the axes so the plotted dimensions come first
        transpose_order = list(range(4))
        transpose_order[plot_dims[0]], transpose_order[0] = 0, plot_dims[0]
        transpose_order[plot_dims[1]], transpose_order[1] = 1, plot_dims[1]
        
        # Transpose the value function
        value_function_transposed = np.transpose(value_function, transpose_order)
        
        # Update coordinates order
        coords_transposed = [coords[i] for i in transpose_order]
        min_bounds_transposed = [min_bounds[i] for i in transpose_order]
        
        # Extract slice again with new ordering
        slice_indices_transposed = [slice(None)] * 4
        for i in range(4):
            if i not in [0, 1]:  # These are now our plot dimensions
                # Find what the original dimension was for this position
                orig_dim = transpose_order.index(i)
                if orig_dim in fixed_indices:
                    slice_indices_transposed[i] = fixed_indices[orig_dim]
        
        slice_2d = value_function_transposed[tuple(slice_indices_transposed)]
        slice_2d_clipped = np.clip(slice_2d, vmin, vmax)
        x_coords = coords_transposed[0]
        y_coords = coords_transposed[1]
    else:
        x_coords = coords[0]
        y_coords = coords[1]
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create labels based on dimension names
    dim_names = ['x', 'y', 'v', 'θ']
    x_label = dim_names[plot_dims[0]]
    y_label = dim_names[plot_dims[1]]
    
    # Create title with fixed values
    title_parts = []
    for dim in range(4):
        if dim not in plot_dims:
            dim_value = fixed_values.get(dim, coords[dim][fixed_indices.get(dim, 0)])
            title_parts.append(f"{dim_names[dim]}={dim_value:.2f}")
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    # Plot contour lines instead of filled contours
    contour = ax.contour(X, Y, slice_2d_clipped.T, levels=contour_levels, 
                        cmap="rainbow", linewidths=contour_linewidth, 
                        linestyles=contour_linestyle, vmin=vmin, vmax=vmax,
                        zorder=1)
    
    # Add contour value labels if requested
    if show_contour_labels:
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot goal circle boundary if provided
    if goal_center is not None and goal_radius is not None:
        goal_circle = plt.Circle(goal_center, goal_radius, color=goal_color, alpha=1.0, 
                               fill=False, linewidth=linewidth, linestyle=linestyle,
                               label='Goal Region')
        ax.add_patch(goal_circle)
    
    # Plot obstacle boundaries if provided
    if obstacles is not None:
        for i, (x_min, x_max, y_min, y_max) in enumerate(obstacles):
            width = x_max - x_min
            height = y_max - y_min
            obstacle_rect = plt.Rectangle((x_min, y_min), width, height, 
                                        color=obstacle_color, alpha=1.0, 
                                        fill=False, linewidth=linewidth, linestyle=linestyle,
                                        label='Obstacle' if i == 0 else "")
            ax.add_patch(obstacle_rect)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Value Function Contour Lines ({", ".join(title_parts)})\nValues clipped to [{vmin}, {vmax}]')
    ax.grid(True, alpha=0.3)
    
    # Set aspect ratio to equal and adjust limits
    ax.set_aspect('equal')
    
    if save_dir is not None:
        fig.savefig(f"{save_dir}/TTR_contour_lines.png", dpi=300, bbox_inches='tight')
        print(f"##### The figure is saved as: {save_dir}/TTR_contour_lines.png")
    
    return fig, ax