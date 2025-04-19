import numpy as np
import matplotlib.pyplot as plt

def read_vector_file(filename, delimiter='\n'):
    """Read vector data from file exported by Eigen's exportVectorToFile"""
    with open(filename, 'r') as f:
        content = f.read()
    # Handle different delimiters (including newlines)
    if delimiter == '\n':
        values = [float(x) for x in content.split()]  # split on any whitespace
    else:
        values = [float(x) for x in content.split(delimiter)]
    return np.array(values)

def plot_vector(filename, delimiter='\n', title='Vector Plot'):
    """Read and plot vector data"""
    y = read_vector_file(filename, delimiter)
    x = np.arange(len(y))  # element indices as X values
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'b-o', markersize=4, linewidth=1, label='Vector values')
    
    plt.title(title)
    plt.xlabel('Element Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # For a file exported with default delimiter (newline)
    plot_vector('Exports/EigenVector/V_Y_FONT.csv', title='My Vector Data')
    
    # For a file exported with comma delimiter
    # plot_vector('vector_data.csv', delimiter=',', title='Comma-separated Vector Data')