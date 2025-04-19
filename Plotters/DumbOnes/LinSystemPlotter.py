import numpy as np
import matplotlib.pyplot as plt

def read_matrix_from_file(filename):
    """
    Reads a matrix from a file.
    Assumes the file has one row per line, with values separated by spaces or commas.
    """
    with open(filename, 'r') as file:
        matrix = []
        for line in file:
            # Split the line by spaces or commas and convert to float
            row = [float(x) for x in line.replace(',', ' ').split()]
            matrix.append(row)
    return np.array(matrix)

def plot_matrix_with_diagonals(matrix):
    """
    Plots the matrix with diagonals highlighted and annotations for non-zero values.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the matrix as a heatmap
    cax = ax.imshow(matrix, cmap='viridis', origin='upper')

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Matrix Values')

    ## Highlight the main diagonal
    #for i in range(matrix.shape[0]):
    #    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
    ## Highlight the anti-diagonal
    #for i in range(matrix.shape[0]):
    #    ax.add_patch(plt.Rectangle((matrix.shape[1] - i - 1.5, i - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))

     #Add annotations for non-zero values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:  # Only annotate non-zero values
                ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center', color='red', fontsize=8)

    # Add labels for rows and columns
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))

    # Add gridlines
    ax.grid(which='both', color='black', linestyle='--', linewidth=0.5)

    # Add titles and labels
    ax.set_title('Matrix Visualization with Diagonals Highlighted')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    import sys



    # Read the matrix from the file
    filename = "Exports/EigenVector/U_X_MATRIX.csv"
    matrix = read_matrix_from_file(filename)

    # Plot the matrix
    plot_matrix_with_diagonals(matrix)