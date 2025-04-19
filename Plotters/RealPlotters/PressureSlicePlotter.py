import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
from matplotlib.colors import Normalize

# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

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

def plot_colormesh_slice(matrix, output_dir="Plots/SliceVisualization", slice_pos=None, variable_name="P"):
    """
    Plot colormesh visualization of a slice of the matrix.
    
    Parameters:
    matrix -- The solution matrix to visualize
    output_dir -- Directory to save the output plot
    slice_pos -- Position to slice along Z axis (if None, middle is used)
    variable_name -- Name of the variable being visualized (e.g., P, U, V, W)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the middle slice along Z axis (k) if not specified
    if slice_pos is None:
        z_middle = matrix.shape[2] // 2
    else:
        z_middle = slice_pos
    
    # Extract the slice
    slice_data = matrix[:, :, z_middle]
    
    # Create X and Y meshgrid for colormesh plot
    X, Y = np.meshgrid(
        np.linspace(0,5,slice_data.shape[1]),  # X-axis (j)
        np.linspace(0,1,slice_data.shape[0])   # Y-axis (i)
    )

    
    # Create figure with high quality settings
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormesh plot

    X = X[1:-2,1:-10]
    Y = Y[1:-2,1:-10]
    slice_data = slice_data[1:-2,1:-10]
    slice_data = slice_data / np.max(slice_data)
    im = ax.pcolormesh(
        X, Y, slice_data,
        cmap='rainbow',
        shading='auto'
    )
    plt.grid(True)
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(f'Visualization of Pressure Field - Normalized\nZ = 0.5', 
                fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'Pressure', fontsize=14)
    
    # Add grid dimensions to the plot
    plt.figtext(0.5, 0.01, f"Grid Size: {matrix.shape[0]}×{matrix.shape[1]}×{matrix.shape[2]}", 
               ha='center', fontsize=12, fontstyle='italic')
    


    # Tight layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save the figure with high resolution
    output_file = os.path.join(output_dir, f"{variable_name}_field_slice_{z_middle}.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_file}")
    
    return fig

def main(input_file, output_dir="Plots/SliceVisualization", slice_pos=None, variable_name="P"):
    """Main function to read data and generate plot."""
    try:
        # Read the matrix from file
        matrix = read_csv_and_reconstruct(input_file)
        
        print(f"Matrix Shape (i=Y, j=X, k=Z): {matrix.shape}")
        
        # Plot the colormesh visualization
        fig = plot_colormesh_slice(matrix, output_dir, slice_pos, variable_name)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a solution field with colormesh.')
    parser.add_argument('--input', type=str, default="Exports/STEP/32_re389/Fields/Field_2254P.csv",
                        help='Path to the solution CSV file')
    parser.add_argument('--output', type=str, default="Plots/SliceVisualization",
                        help='Directory to save the output plot')
    parser.add_argument('--slice', type=int, default=None,
                        help='Position to slice along Z axis (default: middle slice)')
    parser.add_argument('--var', type=str, default="P",
                        help='Name of the variable being visualized (P, U, V, W, etc.)')
    
    args = parser.parse_args()
    main(args.input, args.output, args.slice, args.var)