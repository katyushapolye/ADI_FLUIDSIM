//this will solve the diffusion for u, v and w components
//it works very similarly to the pressure solver, we create a stencil mask for each component at each node and assemble a global matrix from it
//this makes it easier for me to properly handle boundary conditions on free surface problems.
//this implements a default solver to test the diffusion
//after this, there will be a proper ADI implementation, but the update mask should be reusable
#include "../Core/MAC.h"
#include "../Core/Functions.h"
#include "../Core/Definitions.h"
#include "Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>  
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <omp.h>
#ifndef DIFFUSION_SOLVER3D_H
#define DIFFUSION_SOLVER3D_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;
using Eigen::Sizes;
using Eigen::Sparse;
using Eigen::ConjugateGradient;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;

class DiffusionSolver3D{
private:
    static int Nx, Ny, Nz;
    static int NON_ZERO;
    static double dt;
    
    static int* u_collums;
    static int* u_rows;
    static double* u_values;
    
    static int* v_collums;
    static int* v_rows;
    static double* v_values;
    
    static int* w_collums;
    static int* w_rows;
    static double* w_values;
    
    //and the CSR representation
    static SparseMatrix U_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix V_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix W_DIFFUSION_MATRIX_EIGEN;
    
    static CSRMatrix* U_DIFFUSION_MATRIX;
    static CSRMatrix* V_DIFFUSION_MATRIX;
    static CSRMatrix* W_DIFFUSION_MATRIX;
    
    static AMGXSolver* AMGX_Handle;
    
    //indexing grid we use that mirrors the mac, for ease of calculation
    static VectorXd U_IDP;
    static VectorXd V_IDP;
    static VectorXd W_IDP;
    
    static std::vector<Eigen::Tensor<double,3>> U_MASK; //(Nx+1) * Ny * Nz
    static std::vector<Eigen::Tensor<double,3>> V_MASK; //Nx * (Ny+1) * Nz
    static std::vector<Eigen::Tensor<double,3>> W_MASK; //Nx * Ny * (Nz+1)
    
    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

public:
    static void InitializeDiffusionSolver(MAC* grid); 
    static void UpdatePressureMatrix(MAC* grid); //For velocity, we have the opposite of pressure
                                                  //we have dirichlet = 0 at solid walls and
                                                  //grad = 0 at empty boundaries
                                                  //we do need to be extra careful because of the staggering of the grid at corners
    static void SolveDiffusion_Eigen(MAC* gridAnt, MAC* gridSol);
    static void SolveDiffusion_AMGX(MAC* gridAnt, MAC* gridSol);

private:
    static void Update_U_DiffusionMatrix(MAC* grid);           
    static void Update_V_DiffusionMatrix(MAC* grid);      
    static void Update_W_DiffusionMatrix(MAC* grid);  

    static void SolveDiffusion_U_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_W_Eigen(MAC* gridAnt,MAC* gridSol);

    static inline Tensor<double,3> GetUMask(int i ,int j, int k){ 
        return U_MASK[i * ((Nx+1)*Nz)  + (j*Nz)  + k ];
    }
    
    static inline void SetUMask(int i, int j, int k, Tensor<double,3> value){
        U_MASK[i * ((Nx+1)*Nz)  + (j*Nz)  + k ] = value;
    }
    
    static inline int Get_U_IDP(int i, int j, int k){ 
        return U_IDP(i * ((Nx+1)*Nz) + (j*Nz) + k);
    }
    
    static inline void Set_U_IDP(int i, int j, int k, int value){
        U_IDP(i * ((Nx+1)*Nz) + (j*Nz) + k) = value;
    }
    
    // V component accessors: Nx * (Ny+1) * Nz
    // Linearization: i * ((Nx)*Nz) + (j*Nz) + k
    static inline Tensor<double,3> GetVMask(int i, int j, int k){ 
        return V_MASK[i * ((Nx)*Nz)  + (j*Nz)  + k ];
    }
    
    static inline void SetVMask(int i, int j, int k, Tensor<double,3> value){
        V_MASK[i * ((Nx)*Nz)  + (j*Nz)  + k ] = value;
    }
    
    static inline int Get_V_IDP(int i, int j, int k){ 
        return V_IDP(i * ((Nx)*Nz)  + (j*Nz)  + k );
    }
    
    static inline void Set_V_IDP(int i, int j, int k, int value){
        V_IDP(i * ((Nx)*Nz)  + (j*Nz)  + k ) = value;
    }
    
    // W component accessors: Nx * Ny * (Nz+1)
    // Linearization: i * ((Nx)*(Nz+1)) + (j*(Nz+1)) + k
    static inline Tensor<double,3> GetWMask(int i, int j, int k){ 
        return W_MASK[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ];
    }
    
    static inline void SetWMask(int i, int j, int k, Tensor<double,3> value){
        W_MASK[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ] = value;
    }
    
    static inline int Get_W_IDP(int i, int j, int k){ 
        return W_IDP(i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k );
    }
    
    static inline void Set_W_IDP(int i, int j, int k, int value){
        W_IDP(i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ) = value;
    }
};

#endif