import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import vtk
from vtk.util import numpy_support
import matplotlib as mpl
import pandas as pd

# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

def get_exp_data(x, RE):
    try:
        exp_data = pd.read_csv('Data/Experimental/Step/expData_re={}_{}.csv'.format(RE, x))
        exp_data.columns = exp_data.columns.str.strip()
        upar = exp_data['x'].to_list()
        yu_exp = exp_data['y'].to_list()
        return upar, yu_exp
    except FileNotFoundError:
        print(f"No experimental data found for RE={RE}, x={x}")
        return None, None

def get_last_vti_file(directory):
    """Find the last VTI file in a directory based on numerical ordering."""
    vti_files = glob.glob(os.path.join(directory, "grid-*.vti"))
    if not vti_files:
        raise FileNotFoundError(f"No VTI files found in {directory}")
    
    # Extract numbers and sort
    def extract_number(f):
        try:
            return int(f.split('-')[-1].split('.')[0])
        except:
            return -1
    
    vti_files.sort(key=extract_number)
    return vti_files[-1]

def read_velocity_from_vti(filename):
    """Read vectorial velocity data from a VTI file."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    data = reader.GetOutput()
    velocity_data = data.GetPointData().GetArray("velocity")
    dimensions = data.GetDimensions()
    
    velocity_array = numpy_support.vtk_to_numpy(velocity_data)
    velocity_array = velocity_array.reshape((dimensions[0] * dimensions[1] * dimensions[2], 3))
    
    u = velocity_array[:, 0].reshape(dimensions, order='F')
    v = velocity_array[:, 1].reshape(dimensions, order='F')
    w = velocity_array[:, 2].reshape(dimensions, order='F')
    
    return u, v, w, dimensions

def get_velocity_profiles(resolution, reynolds_number, factor=20.0):
    """Get velocity profiles for a specific resolution and Reynolds number."""
    # Construct directory path and find last VTI file
    vtk_dir = f"Exports/STEP/{resolution}_re{reynolds_number}/VTK/"
    try:
        last_vti = get_last_vti_file(vtk_dir)
        print(f"Processing {resolution}x{resolution} resolution, Re={reynolds_number}: using {os.path.basename(last_vti)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    # Read velocity data
    U, V, W, dimensions = read_velocity_from_vti(last_vti)
    
    # Print dimensions to understand the domain shape
    print(f"Domain dimensions for resolution {resolution}: {dimensions}")
    
    # For asymmetric domain, adjust the approach
    nx, ny, nz = dimensions[0], dimensions[1], dimensions[2]
    
    # Determine z-midpoint - use 0 for 1D depth
    zMid = 0 if nz == 1 else nz // 2
    print(f"Using z-index: {zMid}")
    
    # Physical domain dimensions
    x_physical_length = 5.0
    y_physical_length = 1.0
    
    # Define absolute x positions for cuts in physical units
    x_physical_cuts = [0.0, 1, 1.53, 2, 3.06, 4.0]
    
    # Calculate cell size in physical units
    dx = x_physical_length / nx
    dy = y_physical_length / ny
    
    # Convert physical positions to indices
    x_indices = [min(max(int(x_pos / dx), 0), nx-1) for x_pos in x_physical_cuts]
    
    # Y positions in physical units
    Y_physical = np.linspace(0, y_physical_length, ny)
    
    # Get velocity profiles at each cut
    profiles = []
    for x_idx, x_pos in zip(x_indices, x_physical_cuts):
        try:
            if nz == 1:
                uVel = (U[x_idx, :, 0] + U[x_idx-1, :, 0])/2.0  # If depth is 1, use index 0
            else:
                uVel = (U[x_idx, :, zMid] + U[x_idx-1, :, zMid])/2.0
            
            profiles.append((x_pos, uVel, Y_physical))
        except IndexError as e:
            print(f"Error at x_idx={x_idx} for resolution {resolution}: {e}")
            profiles.append((x_pos, None, None))
    
    return profiles, factor

def main():
    # Fixed Reynolds number
    reynolds_number = 389
    
    # Multiple resolutions to compare
    resolutions = [16,32, 64]
    
    # Scale factor for velocity
    factor = 80.0
    
    # Define absolute x positions for cuts in physical units
    x_physical_cuts = [0.5, 1, 1.53, 2, 3.06, 4.0]
    
    # Plot configuration
    velocity_xlim = (-factor*0.25, factor + 10.0)  # Set velocity axis limits
    
    # Create figure with multiple subplots
    num_cuts = len(x_physical_cuts)
    fig, axes = plt.subplots(1, num_cuts, figsize=(num_cuts*3, 6), sharey=True)
    
    # Define colors for different resolutions
    colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange', 'purple']
    
    # Legend entries
    legend_entries = []
    
    # Process each resolution
    for i, res in enumerate(resolutions):
        # Get velocity profiles
        profiles_data = get_velocity_profiles(res, reynolds_number, factor)
        
        if profiles_data is None:
            print(f"Skipping resolution {res} due to errors")
            continue
            
        profiles, factor = profiles_data
        
        
        # Plot each profile at its corresponding x position
        for j, (x_pos, uVel, Y_physical) in enumerate(profiles):

            if uVel is not None and Y_physical is not None:

                
                line, = axes[j].plot(uVel*factor, Y_physical*10, color=colors[i], 
                                     linewidth=2, label=f'Resolution {res}x{res}')
                
                # Add experimental data if available at x=1.53
                if (x_pos == 1.53 or x_pos == 3.06 )and i == 0:  # Only add experimental data once
                    exp_x, exp_y = get_exp_data(x_pos, reynolds_number)
                    if exp_x and exp_y:
                        axes[j].scatter(exp_x, exp_y, color='black', marker='o', 
                                        s=75, alpha=0.7, label=f'Experimental')
            
            # Only add to legend once
            if j == 0:
                legend_entries.append(line)
    
    # Configure each subplot
    for i, (ax, x_pos) in enumerate(zip(axes, x_physical_cuts)):
        ax.set_box_aspect(3.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(velocity_xlim)
        ax.set_title(f'x = {x_pos:.2f}', fontsize=12)
        
        if i == 0:
            ax.set_ylabel('y-position', fontsize=12)
        
        ax.set_xlabel('u-velocity', fontsize=12)
    
    # Add legend to the figure
    fig.legend(legend_entries + ([plt.Line2D([0], [0], color='black', marker='o', linestyle='', markersize=5)] 
                                if any(exp_data[0] for exp_data in [get_exp_data(1.53, reynolds_number)]) else []),
              ([f'Resolution {res}x{res}' for res in resolutions if get_velocity_profiles(res, reynolds_number)] + 
               ['Experimental'] if any(exp_data[0] for exp_data in [get_exp_data(1.53, reynolds_number)]) else []),
              loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(resolutions)+1, frameon=True)
    
    # Add a main title
    fig.suptitle(f'Backwards Facing Step Velocity Profiles for Re={reynolds_number}\nComparing Different Grid Resolutions', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save figure
    plt.savefig(f"step_flow_resolution_comparison_re{reynolds_number}.png", dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()