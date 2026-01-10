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
#plt.style.use('seaborn-v0_8-darkgrid')
#mpl.rcParams['font.family'] = 'serif'

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


re_1000 = np.array([
    0.9651474530831099, 0.7238605898123325, 0.549597855227882, 0.386058981233244,
    0.23592493297587125, 0.16890080428954435, 0.13672922252010733, 0.1072386058981234,
    0.0884718498659518, 0.06166219839142095, 0.05093833780160861, 0.03217158176943702,
    0.008042895442359255, -0.024128686327077764, -0.05630026809651478, -0.08847184986595169,
    -0.11796246648793562, -0.17158176943699732, -0.22520107238605902, -0.24932975871313667,
    -0.23056300268096508, -0.18766756032171583, -0.12868632707774796, -0.07506702412868627,
    -0.008042895442359255
])

y_exp = np.array([
    0.9950900163666122, 0.9836333878887071, 0.9738134206219313, 0.9590834697217676,
    0.9345335515548282, 0.9132569558101473, 0.8707037643207857, 0.8183306055646482,
    0.7594108019639935, 0.707037643207856, 0.6448445171849427, 0.5761047463175123,
    0.49918166939443537, 0.4124386252045827, 0.3567921440261866, 0.3060556464811784,
    0.27168576104746317, 0.2274959083469722, 0.1783960720130933, 0.12765957446808512,
    0.0851063829787234, 0.058919803600654665, 0.03600654664484452, 0.016366612111292964,
    0.0032733224222585926
])


v_vel_1000  = np.array([0.002680965147453083, 0.021447721179624665, 0.0388739946380697, 0.06836461126005362, 0.1032171581769437, 0.15013404825737264, 0.19839142091152814, 0.257372654155496, 0.32707774798927614, 0.38069705093833783, 0.4517426273458445, 0.5361930294906166, 0.6005361930294906, 0.6487935656836461, 0.7024128686327078, 0.757372654155496, 0.8002680965147453, 0.8498659517426274, 0.8954423592493298, 0.9276139410187668, 0.9718498659517426, 0.9892761394101877])
y_pos_1000 = np.array([0.003278688524590123, 0.08852459016393444, 0.16393442622950816, 0.19999999999999996, 0.22622950819672139, 0.20655737704918042, 0.17377049180327875, 0.14754098360655732, 0.09836065573770503, 0.06885245901639347, 0.042622950819672045, 0.009836065573770592, -0.016393442622950838, -0.042622950819672156, -0.059016393442622994, -0.1081967213114754, -0.16393442622950816, -0.2786885245901639, -0.39344262295081966, -0.3868852459016393, -0.19016393442622948, -0.06885245901639347])

v_vel_100 = np.array([0.0035842293906810036, 0.04540023894862604, 0.09318996415770608, 0.14695340501792115, 0.2031063321385902, 0.26642771804062126, 0.3261648745519713, 0.3763440860215054, 0.4348864994026284, 0.4994026284348865, 0.5388291517323776, 0.5794504181600956, 0.6296296296296295, 0.6762246117084826, 0.7275985663082437, 0.7801672640382318, 0.8327359617682197, 0.8912783751493428, 0.9307048984468339, 0.9605734767025089, 0.9976105137395459])
y_pos_100 = np.array([-0.002915451895043719, 0.061224489795918435, 0.10787172011661816, 0.13994169096209919, 0.14577259475218662, 0.13994169096209919, 0.11953352769679304, 0.09037900874635563, 0.055393586005831, 0.011661807580174877, -0.026239067055393583, -0.06705539358600587, -0.11661807580174932, -0.16326530612244894, -0.21574344023323622, -0.24781341107871724, -0.24781341107871724, -0.2011661807580175, -0.1428571428571429, -0.09329446064139946, -0.023323615160349864])

def get_plot_styles(style_id):
    """Return color, line style, font, and grid combinations based on style_id (1-5)."""
    style_schemes = {
        1: {
            'colors': ['mediumorchid', 'gold', 'royalblue', 'forestgreen', 'firebrick'],
            'line_styles': ['-', '--', ':', '-.', '-'],
            'font_family': 'serif',
            'font_size': 14,
            'title_size': 16,
            'grid_style': '--',
            'grid_alpha': 0.7,
            'grid_color': 'gray'
        },
        2: {
            'colors': ['crimson', 'darkorange', 'darkgreen', 'navy', 'purple'],
            'line_styles': ['-', '-', '-', '-', '-'],
            'font_family': 'sans-serif',
            'font_size': 12,
            'title_size': 14,
            'grid_style': '-',
            'grid_alpha': 0.3,
            'grid_color': 'lightgray'
        },
        3: {
            'colors': ['black', 'gray', 'dimgray', 'darkgray', 'lightgray'],
            'line_styles': ['-', '--', ':', '-.', '-'],
            'font_family': 'monospace',
            'font_size': 13,
            'title_size': 15,
            'grid_style': ':',
            'grid_alpha': 0.5,
            'grid_color': 'black'
        },
        4: {
            'colors': ['red', 'blue', 'green', 'orange', 'magenta'],
            'line_styles': ['--', '--', '--', '--', '--'],
            'font_family': 'sans-serif',
            'font_size': 18,
            'title_size': 18,
            'grid_style': '--',
            'grid_alpha': 0.4,
            'grid_color': 'blue'
        },
        5: {
            'colors': ['teal', 'coral', 'gold', 'orchid', 'lightseagreen'],
            'line_styles': [':', '-.', '-', '--', ':'],
            'font_family': 'serif',
            'font_size': 16,
            'title_size': 20,
            'grid_style': '-.',
            'grid_alpha': 0.6,
            'grid_color': 'darkgreen'
        }
    }
    
    if style_id not in style_schemes:
        print(f"Warning: style_id {style_id} not found. Using default style 1.")
        style_id = 1
    
    return style_schemes[style_id]

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

def main(style_id=1):
    # Define resolutions to compare
    resolutions = [8, 16, 32, 64, 128]
    
    # Get style configuration based on style_id
    style_config = get_plot_styles(style_id)
    colors = style_config['colors']
    line_styles = style_config['line_styles']
    
    # Set font properties
    plt.rcParams['font.family'] = style_config['font_family']
    plt.rcParams['font.size'] = style_config['font_size']
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Configure first axis (for U velocity) with custom grid
    ax1.grid(True, linestyle=style_config['grid_style'], 
             alpha=style_config['grid_alpha'], color=style_config['grid_color'])
    ax1.set_xlim(-1, 1)  # X-axis from -1 to 1 (for U velocity)
    ax1.set_ylim(0, 1)   # Y-axis from 0 to 1 (for U velocity)
    ax1.set_xlabel('u velocity', fontsize=style_config['font_size'])
    ax1.set_ylabel('y position', fontsize=style_config['font_size'], color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create second x-axis at top for V velocity
    ax2 = ax1.twiny()
    ax2.set_xlim(0, 1)   # X-axis from 0 to 1 (for position in V velocity)
    ax2.set_xlabel('x position', fontsize=style_config['font_size'])
    
    # Create second y-axis on the right for V velocity
    ax3 = ax1.twinx()
    ax3.set_ylim(-1, 1)  # Y-axis from -1 to 1 (for V velocity)
    ax3.set_ylabel('v velocity', fontsize=style_config['font_size'], color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    
    # Plot experimental data
    ax1.scatter(re_1000, y_exp, facecolor='none', marker='s',s=80,
                label='Yang et al.', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.scatter( 2*v_vel_1000-1,2.0*y_pos_1000, facecolor='none', marker='s', s=80, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Read and plot each resolution
    for res, color, ls in zip(resolutions, colors, line_styles):
        res = 150
        try:
            # Construct directory path and find last VTI file
            vtk_dir = f"Exports/CAVITY/{res}_2D_re100/VTK/"
            last_vti = get_last_vti_file(vtk_dir)
            print(f"Processing {res}x{res} resolution: using {os.path.basename(last_vti)}")
            
            U, V, W = read_velocity_from_vti(last_vti)
            
            # Grid setup
            N = U.shape[0]
            dh = 1.0/N
            zMid = N//2
            xMid = N//2
            Y = np.linspace(dh/2, 1.0-dh/2, N)
            X = np.linspace(dh/2, 1.0-dh/2, N)
            
            uVel = U[xMid,:,zMid]
            vVel = V[:,xMid,zMid]
            
            # Plot U velocity profile (using left/bottom axes)
            #ax1.set_aspect(2.0,adjustable='datalim')
            #ax2.set_aspect(1.0,adjustable='datalim')
            ax3.set_aspect(0.5,adjustable='datalim')

            line_u = ax1.plot(uVel, Y, color=color, linewidth=1.5,
                    label=f'{res}³', linestyle=ls)
            
            # Plot V velocity profile (using top/right axes)
            # Here we use ax3 for y-axis (-1 to 1) and X for x-position (0 to 1)
            line_v = ax3.plot(2*X-1, 2.0*vVel, color=color, linewidth=1.5,
                    linestyle=ls)
            
            # Calculate and print statistics
            uVel_interp = np.interp(y_exp, Y, uVel)
            stats = calculate_statistics(uVel_interp, re_1000)
            print(f"Statistics for {res}x{res} resolution:")
            for key, value in stats.items():
                print(f" {key}: {value:.6f}")
                
        except Exception as e:
            print(f"Error processing resolution {res}: {str(e)}")
            continue
    
    # Set plot properties
    #ax1.set_title(f'Cavity Flow Velocity Profile at Re=1000\nComparison of Different Resolutions',
    #             fontsize=style_config['title_size'], fontweight='bold', pad=20)
    
    # Create combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    
    
    # Add style indicators to the legend
    u_style = plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle=line_styles[0])
    v_style = plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle=line_styles[0])
    
    ax1.legend(lines_1 + [u_style, v_style],
               labels_1, 
               loc='lower left', frameon=True, framealpha=0.9, fontsize=style_config['font_size'])
    
    # Save and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--style', type=int, default=4, choices=[1, 2, 3, 4, 5], 
                        help="Plot style scheme (1-5): 1=mixed colors/styles, 2=solid colored lines, 3=grayscale with different styles, 4=colored dashed lines, 5=vibrant colors with mixed styles")
    args = parser.parse_args()
    main(args.style)