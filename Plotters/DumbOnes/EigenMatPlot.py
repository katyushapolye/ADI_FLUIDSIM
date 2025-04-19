import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def read_sparse_matrix_from_file(filename):
    with open(filename, 'rb') as f:
        # Read matrix dimensions
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        nnz = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read COO data
        coo_data = np.fromfile(f, dtype=np.dtype([
            ('row', np.int32),
            ('col', np.int32),
            ('data', np.float64)
        ]), count=nnz)
        
        # Create COO matrix
        row = coo_data['row']
        col = coo_data['col']
        data = coo_data['data']
        
        return coo_matrix((data, (row, col)), shape=(rows, cols))

def plot_sparse_matrix(filename):
    # Read the sparse matrix
    sparse_mat = read_sparse_matrix_from_file(filename)
    
    # Convert to dense format for visualization
    dense_mat = sparse_mat.toarray()
    
    # Plot the matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(dense_mat, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Sparse Matrix Visualization (Dense Form)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_sparse_matrix("Exports/Eigen/PressureMat.txt")