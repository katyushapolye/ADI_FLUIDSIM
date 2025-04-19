import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.colors import Normalize

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

def exactVel(x, y, z, t):
    u = np.cos(x)*np.sin(y)*np.cos(z) * np.e**(-2*t)
    v = -np.sin(x)*np.cos(y)*np.sin(z)* np.e**(-2*t)
    w = 0.0* np.e**(-2*t)
    return u, v, w

def main(t):
    os.makedirs("Plots/TaylorGreenExactSlices", exist_ok=True)
    
    # Create grid




    u = read_csv_and_reconstruct("Exports/STEP/16_re389/Fields/Field_411U.csv")
    v = read_csv_and_reconstruct("Exports/STEP/16_re389/Fields/Field_411V.csv")
    w = read_csv_and_reconstruct("Exports/STEP/16_re389/Fields/Field_411W.csv")

    z = np.linspace(0, 1, u.shape[0])
    X, Y, Z = np.meshgrid(
        np.linspace(0, 5, u.shape[1]-1),
        np.linspace(0, 1, u.shape[0]),
        np.linspace(0, 1, u.shape[2])
    )
    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    #interp to center , ijk sintax
    U = np.zeros((u.shape[0],u.shape[1]-1,u.shape[2]))
    V = np.zeros((v.shape[0]-1,v.shape[1],v.shape[2]))
    W = np.zeros((w.shape[0],w.shape[1],w.shape[2]-1))

    U[:,:,:] = (u[:,1:,:] + u[:,:-1,:]) /2.0
    V[:,:,:] = (v[1:,:,:] + v[:-1,:,:]) /2.0
    W[:,:,:] = (w[:,:,1:] + w[:,:,:-1]) /2.0
    
    # Calculate magnitude range (FIXED SCALE)
    
    norm = Normalize(vmin=0, vmax=1.0)  # Fixed scale from 0 to max
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, width_ratios=[1,1,1,0.05])
    fig.suptitle(f'Taylor-Green Vortex at t={t:.2f}', y=1.02, fontsize=20)
    

    # Get slice data
    X_slice = X[:, :, 4]
    Y_slice = Y[:, :, 4]
    U_slice = U[:, :, 4]
    V_slice = V[:, :, 4]
    magnitude = np.sqrt(U_slice**2 + V_slice**2)
    
    # Plot with FIXED color scale
    plt.quiver(X_slice, Y_slice, U_slice, V_slice, magnitude,
                     scale=100, width=0.007, cmap='viridis', norm=norm)
    
    plt.title(f"Middle Z slice", pad=12)
    plt.xlim((0, 5))


    # Add colorbar with FIXED scale

    
    plt.tight_layout()
    #plt.savefig(f"Plots/TaylorGreenExactSlices/TGV_t={t:.2f}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=float, default=0.0)
    args = parser.parse_args()
    main(args.t)