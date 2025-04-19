import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import re
import argparse

# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

def process_data_files(directory):
    """
    Process all CSV files in the given directory, extracting average computation times
    for different algorithms and problem sizes.
    """
    # Dictionary to store data for each type and size
    adi_data = {}
    pressure_data = {}
    
    # Get all csv files in the directory
    files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Extract data from files
    for file_path in files:
        # Extract file name for processing
        file_name = os.path.basename(file_path)
        
        # Use regex to extract type and size number
        match = re.search(r'time_(\w+)_(\d+)\.csv', file_name)
        if match:
            file_type = match.group(1)  # 'adi' or 'pressure'
            size = int(match.group(2))  # numerical size value
            
            # Read the values, assuming they are in the format "value,"
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Convert to float, removing the trailing comma
                values = [float(line.strip().rstrip(',')) for line in lines if line.strip()]
                
                # Calculate average
                avg = np.mean(values)
                
                # Store data in appropriate dictionary
                if file_type == 'adi':
                    adi_data[size] = avg
                elif file_type == 'pressure':
                    pressure_data[size] = avg
                
                print(f"File: {file_name}, Type: {file_type}, Size: {size}, Average: {avg:.6e}")
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Skipping file with unrecognized format: {file_name}")
    
    return adi_data, pressure_data, len(files)

def create_plot(adi_data, pressure_data, output_file="computation_time_comparison.png"):
    """
    Create and save a plot comparing computation times with contribution percentages.
    """
    # Sort data by size for plotting
    adi_sizes = sorted(adi_data.keys())
    pressure_sizes = sorted(pressure_data.keys())
    adi_averages = [adi_data[size] for size in adi_sizes]
    pressure_averages = [pressure_data[size] for size in pressure_sizes]
    
    # Create the plot with better proportions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot data with improved styling
    if adi_data:
        ax.plot(adi_sizes, adi_averages, 'o-', color='royalblue', linewidth=2.5, 
                markersize=8, label='ADI')
        
        # Add value annotations for ADI
        for i, size in enumerate(adi_sizes):
            ax.annotate(f"{adi_averages[i]:.2e}", 
                      xy=(size, adi_averages[i]), 
                      xytext=(0, 10),
                      textcoords='offset points',
                      ha='center', 
                      fontsize=9)
    
    if pressure_data:
        ax.plot(pressure_sizes, pressure_averages, 's-', color='forestgreen', 
               linewidth=2.5, markersize=8, label='Pressure')
        
        # Add value annotations for Pressure
        for i, size in enumerate(pressure_sizes):
            ax.annotate(f"{pressure_averages[i]:.2e}", 
                      xy=(size, pressure_averages[i]), 
                      xytext=(0, -15),
                      textcoords='offset points',
                      ha='center', 
                      fontsize=9)
    
    # Calculate and display contribution percentages where both measurements exist
    common_sizes = sorted(set(adi_sizes) & set(pressure_sizes))
    for size in common_sizes:
        total_time = adi_data[size] + pressure_data[size]
        adi_percent = (adi_data[size] / total_time) * 100
        pressure_percent = (pressure_data[size] / total_time) * 100
        
        # Add contribution text near the midpoint between the two points
        midpoint_y = 0.2+ (adi_data[size] + pressure_data[size]) / 2
        contribution_text = f"ADI: {adi_percent:.1f}%\nPressure: {pressure_percent:.1f}%"
        
        ax.annotate(contribution_text,
                  xy=(size, midpoint_y),
                  xytext=(0, 0),
                  textcoords='offset points',
                  ha='center',
                  va='center',
                  fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Improve x-axis display for grid sizes
    if adi_sizes or pressure_sizes:
        all_sizes = sorted(set(adi_sizes + pressure_sizes))
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([f"{size}³" for size in all_sizes])
    
    # Set labels with better fonts
    ax.set_xlabel('Grid Size', fontsize=14)
    ax.set_ylabel('Average Computation Time (s)', fontsize=14)
    
    # Add a descriptive title
    ax.set_title('Average Computation Time by Algorithm and Problem Size', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Format y-axis in scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Improve legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure with high resolution
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def main(directory="Data/Times/Re100", output_file="computation_time_comparison.png"):
    # Process data files
    adi_data, pressure_data, num_files = process_data_files(directory)
    
    # Create and save the plot
    fig = create_plot(adi_data, pressure_data, output_file)
    
    # Show the plot
    plt.show()
    
    print(f"Processed {num_files} files.")
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare computation times for ADI and Pressure algorithms.')
    parser.add_argument('--directory', type=str, default="Data/Times/Re100",
                        help='Directory containing the time CSV files')
    parser.add_argument('--output', type=str, default="computation_time_comparison.png",
                        help='Output filename for the plot')
    
    args = parser.parse_args()
    main(args.directory, args.output)