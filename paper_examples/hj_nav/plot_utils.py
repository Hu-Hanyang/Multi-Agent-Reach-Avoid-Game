import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Union

def plot_value_contour(
    grid,  # Grid object with min, max, pts_each_dim attributes
    value_function: np.ndarray,
    plot_dims: List[int] = [0, 1],
    fixed_values: Optional[Dict[int, float]] = None,
    vmin: float = 0,
    vmax: float = 10,
    ax: Optional[plt.Axes] = None,
    save_dir: Optional[str] = None,
    contour_linewidth: float = 2,
    contour_linestyle: str = '-',
    contour_label_fontsize: int = 8,
    show_contour_labels: bool = True,
    contour_levels: int = 20,
    title: Optional[str] = None
):
    """
    Plot contour lines of a 2D or 4D value function.
    
    Args:
        grid: Grid object with min, max, pts_each_dim attributes
        value_function: numpy array of value function values
        plot_dims: Dimensions to plot [x_dim, y_dim]
        fixed_values: Dictionary of fixed values for non-plotted dimensions
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        ax: Matplotlib axes to plot on
        save_dir: Directory to save the figure
        contour_linewidth: Width of contour lines
        contour_linestyle: Style of contour lines
        contour_label_fontsize: Font size for contour labels
        show_contour_labels: Whether to show contour value labels
        contour_levels: Number of contour levels or specific levels
        title: Custom title for the plot
    """
    
    # Get grid information
    min_bounds = grid.min
    max_bounds = grid.max
    resolutions = grid.pts_each_dim
    
    dim = len(resolutions)
    dim_names = ['x', 'y', 'v', 'θ'][:dim]
    coords = [np.linspace(min_bounds[i], max_bounds[i], resolutions[i]) for i in range(dim)]
    
    # Handle 2D case
    if dim == 2:
        if plot_dims != [0, 1]:
            plot_dims = [0, 1]
        if fixed_values is not None:
            fixed_values = None
            
        x_coords, y_coords = coords[0], coords[1]
        slice_2d = value_function
        title_parts = ["2D Value Function"]
        
    # Handle 4D case  
    else:
        if fixed_values is None:
            fixed_values = {2: 0.5, 3: np.pi/2}
        
        fixed_indices = {dim: np.argmin(np.abs(coords[dim] - value)) for dim, value in fixed_values.items()}
        
        slice_indices = [slice(None)] * 4
        for dim, idx in fixed_indices.items():
            slice_indices[dim] = idx
        
        slice_2d = value_function[tuple(slice_indices)]
        
        if plot_dims != [0, 1]:
            transpose_order = list(range(4))
            transpose_order[plot_dims[0]], transpose_order[0] = 0, plot_dims[0]
            transpose_order[plot_dims[1]], transpose_order[1] = 1, plot_dims[1]
            
            value_function_transposed = np.transpose(value_function, transpose_order)
            coords_transposed = [coords[i] for i in transpose_order]
            
            slice_indices_transposed = [slice(None)] * 4
            for i in range(4):
                if i not in [0, 1]:
                    orig_dim = transpose_order.index(i)
                    if orig_dim in fixed_indices:
                        slice_indices_transposed[i] = fixed_indices[orig_dim]
            
            slice_2d = value_function_transposed[tuple(slice_indices_transposed)]
            x_coords, y_coords = coords_transposed[0], coords_transposed[1]
        else:
            x_coords, y_coords = coords[0], coords[1]
        
        # Generate title parts showing fixed values
        title_parts = []
        for dim in range(4):
            if dim not in plot_dims:
                dim_value = fixed_values.get(dim, coords[dim][fixed_indices.get(dim, 0)])
                title_parts.append(f"{dim_names[dim]}={dim_value:.2f}")
    
    # Clip data for display
    slice_2d_clipped = np.clip(slice_2d, vmin, vmax)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    # Create contour plot
    contour = ax.contour(X, Y, slice_2d_clipped.T, levels=contour_levels, 
                        cmap="cividis", linewidths=contour_linewidth, 
                        linestyles=contour_linestyle, vmin=vmin, vmax=vmax)
    
    if show_contour_labels:
        ax.clabel(contour, inline=True, fontsize=contour_label_fontsize, fmt='%.1f')
    
    # Set axis labels
    if dim == 2:
        x_label, y_label = dim_names[0], dim_names[1]
    else:
        x_label, y_label = dim_names[plot_dims[0]], dim_names[plot_dims[1]]
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal')
    
    # Set title
    if title is None:
        if dim == 2:
            title = f'2D Value Function Contour'
        else:
            plot_dims_str = f"({dim_names[plot_dims[0]]}, {dim_names[plot_dims[1]]})"
            fixed_vals_str = ", ".join(title_parts)
            title = f'4D Value Function - Plot dims: {plot_dims_str}\nFixed: {fixed_vals_str}'
    
    ax.set_title(title)
    
    # Save figure if directory provided
    if save_dir is not None:
        dim_suffix = "2d" if dim == 2 else "4d"
        fig.savefig(f"{save_dir}/hj_value_{dim_suffix}_contour.png", dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_2d_value_function(value_function, plot_dim):
    pass

def visualize_human_robot(grid, 
                          value_function,
                          plot_dims,
                          fixed_values,
                          human_position,
                          robot_state, 
                          threshold, 
                          fig,
                          ax):
    pass