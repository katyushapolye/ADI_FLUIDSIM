import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import argparse

# Set up a nice style for the plot
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'

def read_csv_values(filename):
    """
    Read values from a CSV file where each value is on a separate line.
    Returns an array of values.
    """
    values = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace and commas
            value = line.strip().strip(',')
            if value:  # Skip empty lines
                values.append(float(value))
    return np.array(values)

def main(csv_file):
    # Create output directory if it doesn't exist
    
    # Read values from the CSV file
    y_values = read_csv_values(csv_file)
    
    # Create x values (indices)
    x_values = np.arange(1, len(y_values) + 1)
    
    # Create a nice figure with better proportions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot the data
    ax.plot(x_values, y_values,  color='royalblue', linewidth=2.5)
    
    # Set the x-axis limits with some padding
    ax.set_xlim(0.5, len(y_values) + 0.5)
    
    # Set the y-axis limits with some padding
    y_min = min(y_values) * 0.95
    y_max = max(y_values) * 1.05
    ax.set_ylim(y_min, 4e-5)
    
    # Format y-axis in scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Set labels with a better font
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Absolute Divergency Sum', fontsize=14)
    
    # Add a descriptive title
    ax.set_title("Absolute Divergency Sum from Vortex Shedding at Re = 1000.0 and N = 64", fontsize=16, fontweight='bold', pad=20)
    
    # Add min and max annotations
    min_idx = np.argmin(y_values)
    max_idx = np.argmax(y_values)
    
    ax.annotate(f'Min: {y_values[min_idx]:.3e}',
                xy=(x_values[min_idx], y_values[min_idx]),
                xytext=(x_values[min_idx]+0.3, y_values[min_idx]*0.98),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    ax.annotate(f'Max: {y_values[max_idx]:.3e}',
                xy=(x_values[max_idx], y_values[max_idx]),
                xytext=(x_values[max_idx]-0.3, y_values[max_idx]*1.02),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Tight layout
    plt.tight_layout()
    

    
    # Show the plot
    plt.show()

if __name__ == "__main__":


    main("Data/Divergency/Divergency_OBSTACLE_64_re1000.csv")