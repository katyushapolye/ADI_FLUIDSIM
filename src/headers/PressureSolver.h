
#include "MAC.h"
#include "Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>  
#include <Eigen/IterativeLinearSolvers>
#include "Functions.h"

#include <vector>




#ifndef PRESSURE_SOLVER_H
#define PRESSURE_SOLVER_H
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;
using Eigen::Sizes;
using Eigen::Sparse;
using Eigen::ConjugateGradient;

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;
class PressureSolver
{
private:
    static int Nx,Ny,Nz;
    static int NON_ZERO;
    static double dt;

    //Pressure Matrix is stored as a COO sparse matrix for ease of use,
    //non eigen vectors because of cuda
    static int* collums;
    static int* rows;
    static double* values;
    //and the CSR representation
    static SparseMatrix PRESSURE_MATRIX_EIGEN;

    static CSRMatrix* PRESSURE_MATRIX;






    //indexing grid we use that mirrors the mac, for ease of calculation
    //InDexPressure
    static VectorXd IDP;

    static std::vector<Eigen::Tensor<double,3>> PRESSURE_MASK;
    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

    
public:

    static void InitializePressureSolver(MAC* grid,double dt);
    static void UpdatePressureMatrix(MAC* grid, double dt);

    //solves the pressure using the velocity and puts on the grid, debug only, uses eigen to solve on CPU
    static void SolvePressure_EIGEN(MAC* gridAnt);
    static void SolvePressure_AMGX(MAC* gridAnt);

    //takes the pressure on the grid and crrects the velocity
    static void ProjectPressure(MAC* grid);

    static inline Tensor<double,3> GetPressureMask(int i,int j, int k){ return PRESSURE_MASK[i * ((Nx)*(Nz))  + (j*Nz)  + k ];};
    static inline void SetPressureMask(int i,int j, int k,Tensor<double,3> value){PRESSURE_MASK[i * ((Nx)*(Nz))  + (j*Nz)  + k ] = value;};
    static inline int GetIDP(int i,int j, int k){ return IDP(i * ((Nx)*(Nz))  + (j*Nz)  + k );};
    static inline void SetIDP(int i,int j, int k,int value){IDP(i * ((Nx)*(Nz))  + (j*Nz)  + k ) = value;};

    static int GetSolverIterations();



};

#endif


