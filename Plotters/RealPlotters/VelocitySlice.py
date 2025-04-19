import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from matplotlib.colors import Normalize
import matplotlib as mpl
import vtk
from vtk.util import numpy_support



# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

# Experimental data for Re=1000
y_exp = np.array([
    1.0000, 0.9600, 0.9200, 0.8800, 0.8400, 0.8000, 0.7600, 0.7200, 0.6800, 0.6400,
    0.6000, 0.5600, 0.5200, 0.4800, 0.4400, 0.4000, 0.3600, 0.3200, 0.2800, 0.2600,
    0.2400, 0.2000, 0.1600, 0.1200, 0.0800, 0.0400, 0.0000
])

re_100 = np.array([
    1.0000, 0.7159, 0.4704, 0.2841, 0.1516, 0.0577, -0.0119, -0.0660, -0.1097, -0.1452,
    -0.1732, -0.1937, -0.2067, -0.2120, -0.2100, -0.2017, -0.1886, -0.1722, -0.1539, -0.1444,
    -0.1349, -0.1155, -0.0955, -0.0748, -0.0529, -0.0286, 0.0000
])

re_400 = np.array([
    1.0000, 0.5563, 0.2868, 0.1730, 0.1285, 0.1052, 0.0875, 0.0714, 0.0550, 0.0369,
    0.0159, -0.0092, -0.0391, -0.0734, -0.1113, -0.1505, -0.1872, -0.2159, -0.2319, -0.2341,
    -0.2324, -0.2176, -0.1903, -0.1540, -0.1111, -0.0608, 0.0000
])

re_1000 = np.array([
    1.0000, 0.4119, 0.1912, 0.1389, 0.1148, 0.0959, 0.0798, 0.0661, 0.0540, 0.0432,
    0.0328, 0.0224, 0.0120, 0.0015, -0.0099, -0.0242, -0.0435, -0.0701, -0.1063, -0.1284,
    -0.1528, -0.2049, -0.2512, -0.2754, -0.2611, -0.1812, 0.0000
])

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
    
    return u, v, w

def calculate_statistics(sim_data, ref_data):
    mse = np.mean((ref_data - sim_data)**2)
    l2_norm = np.sqrt(np.sum((ref_data - sim_data)**2))
    correlation = np.corrcoef(ref_data, sim_data)[0, 1]
    max_error = np.max(np.abs(ref_data - sim_data))
    
    return {
        'MSE': mse,
        'L2 Norm': l2_norm,
        'Correlation': correlation,
        'Max Error': max_error
    }

def main(t):

    
    # Define resolutions to compare
    resolutions = [8,16,32, 64, 128]
    colors = ['mediumorchid','gold','royalblue', 'forestgreen', 'firebrick']
    line_styles = ['-', '--', ':','',""]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot experimental data
    ax.scatter(re_1000, y_exp, facecolor='none', marker='o', s=80, 
               label='Jiang et al. (1994)', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Read and plot each resolution
    for res, color, ls in zip(resolutions, colors, line_styles):
        try:
            # Construct directory path and find last VTI file
            vtk_dir = f"Exports/CAVITY/{res}_re1000/VTK/"
            last_vti = get_last_vti_file(vtk_dir)
            print(f"Processing {res}x{res} resolution: using {os.path.basename(last_vti)}")
            
            U, V, W = read_velocity_from_vti(last_vti)
            
            # Grid setup
            N = U.shape[0]
            dh = 1.0/N
            zMid = N//2
            xMid = N//2
            
            Y = np.linspace(dh/2, 1.0-dh/2, N)
            uVel = U[xMid,:,zMid]
            
            # Plot simulation data
            ax.plot(uVel, Y, color=color, linewidth=1.5, 
                    label=f'{res}³')
            
            # Calculate and print statistics
            uVel_interp = np.interp(y_exp, Y, uVel)
            stats = calculate_statistics(uVel_interp, re_1000)
            print(f"Statistics for {res}x{res} resolution:")
            for key, value in stats.items():
                print(f"  {key}: {value:.6f}")
                
        except Exception as e:
            print(f"Error processing resolution {res}: {str(e)}")
            continue
    
    # Set plot properties
    ax.set_xlim((-1, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel('u', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Cavity Flow Velocity Profile at Re=1000\nComparison of Different Resolutions', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=12)
    
    # Save and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=float, default=0.0, help="Time value")
    args = parser.parse_args()
    main(args.t)