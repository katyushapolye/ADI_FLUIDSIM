import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
from matplotlib.colors import Normalize

# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

def exactVel(x, y, z, t):
    """Calculate exact velocity components at given coordinates and time."""
    u = np.cos(x) * np.sin(y) * np.cos(z) * np.exp(-2*t)
    v = -np.sin(x) * np.cos(y) * np.sin(z) * np.exp(-2*t)
    w = 0.0 * np.exp(-2*t)
    return u, v, w

def read_csv_and_reconstruct(filename):
    """Read and reconstruct a 3D matrix from a CSV file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract dimensions (first line: i,j,k)
    i, j, k = map(int, lines[0].strip().split(','))
    
    # Read data (remaining lines)
    flattened = np.array([float(line.strip().strip(',')) for line in lines[1:]])
    
    # Reshape to (i,j,k) where i=Y, j=X, k=Z
    mat = flattened.reshape((i, j, k))  # Now mat[i,j,k] corresponds to Y,X,Z
    
    return mat

def plot_colormesh_slices(mat_exact, mat_approx, output_dir="Plots/SliceComparison", slice_pos=None):
    """
    Plot colormesh visualizations of the middle slice of both matrices.
    
    Parameters:
    mat_exact -- The exact solution matrix
    mat_approx -- The approximate solution matrix
    output_dir -- Directory to save the output plot
    slice_pos -- Position to slice along Z axis (if None, middle is used)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the middle slice along Z axis (k) if not specified
    if slice_pos is None:
        z_middle = mat_exact.shape[2] // 2
    else:
        z_middle = slice_pos
    
    slice_exact = mat_exact[:, :, z_middle]
    
    # Handle possible dimension mismatch
    if mat_approx.shape[1] != mat_exact.shape[1]:
        slice_approx = mat_approx[:, :-(mat_exact.shape[1] - mat_approx.shape[1]), z_middle]
    else:
        slice_approx = mat_approx[:, :, z_middle]
    
    # Calculate error and statistics
    error = slice_exact - slice_approx
    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))
    
    # Create X and Y meshgrid for colormesh plot
    X, Y = np.meshgrid(
        np.arange(slice_exact.shape[1]),  # X-axis (j)
        np.arange(slice_exact.shape[0])   # Y-axis (i)
    )
    
    # Create a common colormap range for both plots
    vmin = min(np.min(slice_exact), np.min(slice_approx))
    vmax = max(np.max(slice_exact), np.max(slice_approx))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure with high quality settings
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Exact Solution Plot ---
    im1 = axes[0].pcolormesh(
        X, Y, slice_exact,
        cmap='viridis',
        norm=norm,
        shading='auto'
    )
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('X Position', fontsize=12)
    axes[0].set_ylabel('Y Position', fontsize=12)
    axes[0].set_title(f'Exact Solution\nZ Slice at k={z_middle}', 
                     fontsize=14, fontweight='bold')
    
    # --- Approximate Solution Plot ---
    im2 = axes[1].pcolormesh(
        X, Y, slice_approx,
        cmap='viridis',
        norm=norm,
        shading='auto'
    )
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('X Position', fontsize=12)
    axes[1].set_title(f'Approximate Solution\nZ Slice at k={z_middle}', 
                     fontsize=14, fontweight='bold')
    
    # --- Error Plot ---
    # Using a different colormap for error
    error_norm = Normalize(vmin=-max_error, vmax=max_error)
    im3 = axes[2].pcolormesh(
        X, Y, error,
        cmap='RdBu_r',  # Red-Blue diverging colormap
        norm=error_norm,
        shading='auto'
    )
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('X Position', fontsize=12)
    axes[2].set_title(f'Error (Exact - Approx)\nMax Err: {max_error:.2e}, Mean Err: {mean_error:.2e}', 
                     fontsize=14, fontweight='bold')
    
    # Add colorbars to each plot
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Value', fontsize=12)
    
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Value', fontsize=12)
    
    cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Error', fontsize=12)
    
    # Add a main title to the figure
    fig.suptitle(f'Comparison of Exact and Approximate Solutions\nGrid Size: {mat_exact.shape[0]}x{mat_exact.shape[1]}x{mat_exact.shape[2]}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add statistics text box
    stats_text = (f"Max Error: {max_error:.6e}\n"
                 f"Mean Error: {mean_error:.6e}\n"
                 f"Max Value: {max(np.max(slice_exact), np.max(slice_approx)):.6e}")
    
    # Add a text box with the statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.92, 0.15, stats_text, fontsize=12,
               verticalalignment='top', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for the title
    
    # Save the figure with high resolution
    output_file = os.path.join(output_dir, f"colormesh_comparison_slice_{z_middle}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_file}")
    
    return fig

def main(exact_file, approx_file, output_dir="Plots/SliceComparison", slice_pos=None):
    """Main function to read data and generate plots."""
    # Read both matrices
    try:
        mat_exact = read_csv_and_reconstruct(exact_file)
        mat_approx = read_csv_and_reconstruct(approx_file)
        
        print(f"Exact Matrix Shape (i=Y, j=X, k=Z): {mat_exact.shape}")
        print(f"Approx Matrix Shape (i=Y, j=X, k=Z): {mat_approx.shape}")
        
        # Calculate max absolute difference
        # Handle potential size differences
        min_i = min(mat_exact.shape[0], mat_approx.shape[0])
        min_j = min(mat_exact.shape[1], mat_approx.shape[1])
        min_k = min(mat_exact.shape[2], mat_approx.shape[2])
        
        max_diff = np.max(np.abs(
            mat_exact[:min_i, :min_j, :min_k] - mat_approx[:min_i, :min_j, :min_k]
        ))
        print(f"Max Absolute Difference: {max_diff:.6e}")
        
        # Plot the colormesh visualizations
        fig = plot_colormesh_slices(mat_exact, mat_approx, output_dir, slice_pos)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and compare exact and approximate solutions.')
    parser.add_argument('--exact', type=str, default="Exports/STEP/64_re389/Fields/Field_50P.csv",
                        help='Path to the exact solution CSV file')
    parser.add_argument('--approx', type=str, default="Exports/STEP/64_re389/Fields/Field_50P.csv",
                        help='Path to the approximate solution CSV file')
    parser.add_argument('--output', type=str, default="Plots/SliceComparison",
                        help='Directory to save the output plot')
    parser.add_argument('--slice', type=int, default=None,
                        help='Position to slice along Z axis (default: middle slice)')
    
    args = parser.parse_args()
    main(args.exact, args.approx, args.output, args.slice)