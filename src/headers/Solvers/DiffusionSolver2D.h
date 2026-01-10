//this will solve the diffusion for u and v components
//it works very similarly to the pressure solver, we create a stencil mas for each component at each node and assemble a global matrix from it
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

#include <omp.h>
#include <vector>



#ifndef DIFFUSION_SOLVER2D_H
#define DIFFUSION_SOLVER2D_H
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;
using Eigen::Sizes;
using Eigen::Sparse;
using Eigen::ConjugateGradient;

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;

class DiffusionSolver2D{

    private:
    static int Nx,Ny;

    static int NON_ZERO;

    static double dt;


    static int* u_collums;
    static int* u_rows;
    static double* u_values;

    static int* v_collums;
    static int* v_rows;
    static double* v_values;

    //and the CSR representation
    static SparseMatrix U_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix V_DIFFUSION_MATRIX_EIGEN;

    static CSRMatrix* U_DIFFUSION_MATRIX;
    static CSRMatrix* V_DIFFUSION_MATRIX;

    static AMGXSolver* AMGX_Handle;



    //indexing grid we use that mirrors the mac, for ease of calculation
    static VectorXd U_IDP;
    static VectorXd V_IDP;

    static std::vector<Eigen::Tensor<double,2>> U_MASK;//Nx+1 Ny
    static std::vector<Eigen::Tensor<double,2>> V_MASK; //Nx Ny+1

    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

public:
    static void InitializeDiffusionSolver(MAC* grid); 
  

    static void SolveDiffusion_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_AMGX(MAC* gridAnt,MAC* gridSol);

    
private:

    static void Update_U_DiffusionMatrix(MAC* grid);           
    static void Update_V_DiffusionMatrix(MAC* grid);      
    static void SolveDiffusion_U_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_Eigen(MAC* gridAnt,MAC* gridSol);

    static inline Tensor<double,2> GetUMask(int i,int j){ return U_MASK[i * ((Nx+1))  + (j)   ];};
    static inline void SetUMask(int i,int j,Tensor<double,2> value){U_MASK[i * ((Nx+1))  + (j)   ] = value;};
    static inline int Get_U_IDP(int i,int j){ return U_IDP(i * ((Nx+1))  + (j)   );};
    static inline void Set_U_IDP(int i,int j,int value){U_IDP(i * ((Nx+1))  + (j)) = value;};


    static inline Tensor<double,2> GetVMask(int i,int j){ return V_MASK[i * ((Nx))  + (j)   ];};
    static inline void SetVMask(int i,int j,Tensor<double,2> value){V_MASK[i * ((Nx))  + (j)   ] = value;};
    static inline int Get_V_IDP(int i,int j){ return V_IDP(i * ((Nx))  + (j)   );};
    static inline void Set_V_IDP(int i,int j,int value){V_IDP(i * ((Nx))  + (j)) = value;};
    
};


#endif
