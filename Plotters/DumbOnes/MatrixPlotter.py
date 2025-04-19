import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_csv_and_reconstruct(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract dimensions (first line: i,j,k)
    i, j, k = map(int, lines[0].strip().split(','))
    
    # Read data (remaining lines)
    flattened = np.array([float(line.strip().strip(',')) for line in lines[1:]])
    
    # Reshape to (i,j,k) where i=Y, j=X, k=Z
    mat = flattened.reshape((i, j, k))  # Now mat[i,j,k] corresponds to Y,X,Z
    
    return mat

def plot_3d_scatter(mat):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate grid coordinates (Y=i, X=j, Z=k)
    Y, X, Z = np.meshgrid(
        np.arange(mat.shape[0]),  # Y axis (i)
        np.arange(mat.shape[1]),  # X axis (j)
        np.arange(mat.shape[2]),  # Z axis (k)
        indexing='ij'  # Use 'ij' indexing to match mat[i,j,k]
    )
    #Y = Y[1:-1,1:-1,1:-1]
    #X = X[1:-1,1:-1,1:-1]
    #Z = Z[1:-1,1:-1,1:-1]
    #mat = mat[1:-1,1:-1,1:-1]

    print("MAX: ", np.max(mat))
    print("MIN: ", np.min(mat))
    
    # Flatten and plot
    scatter = ax.scatter(
        X.flatten(),  # X values (j)
        Z.flatten(),  # Z values (k)
        Y.flatten(),  # Y values (i) -> vertical
        c=mat.flatten(),
        cmap='viridis',
        s=50,  # Marker size
        vmin=np.min(mat),  # Set min of color scale to data min
        vmax=np.max(mat)   # Set max of color scale to data max
    )
    
    # Label axes (Y is upwards)
    ax.set_xlabel('X (j)')
    ax.set_ylabel('Z (k)')
    ax.set_zlabel('Y (i)')
    ax.set_title('3D Scatter Plot (Y=Upwards)')
    
    # Add colorbar with matching scale
    cbar = plt.colorbar(scatter, ax=ax, label='Value')
    
    plt.gca().set_aspect('equal', adjustable='datalim') 
    plt.show()

if __name__ == "__main__":
    # Read and reconstruct matrix (Y=i, X=j, Z=k)
    mat = read_csv_and_reconstruct("Exports/STEP/32_re389/Fields/Field_0SOLID.csv")
    print("Reconstructed Matrix Shape (i=Y, j=X, k=Z):", mat.shape)
    
    # Plot
    plot_3d_scatter(mat)