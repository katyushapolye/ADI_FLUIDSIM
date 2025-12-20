
#include "MAC_2D.h"
#include "Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>  
#include <Eigen/IterativeLinearSolvers>
#include "Functions.h"

#include <vector>




#ifndef PRESSURE_SOLVER2D_H
#define PRESSURE_SOLVER2D_H
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;
using Eigen::Sizes;
using Eigen::Sparse;
using Eigen::ConjugateGradient;

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;
class PressureSolver2D
{
private:
    static int Nx,Ny;
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

    static AMGXSolver* AMGX_Handle;



    //indexing grid we use that mirrors the mac, for ease of calculation
    //InDexPressure
    static VectorXd IDP;

    static std::vector<Eigen::Tensor<double,2>> PRESSURE_MASK;
    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

    
public:

    static void InitializePressureSolver(MAC2D* grid,double dt);
    static void UpdatePressureMatrix(MAC2D* grid, double dt);

    //solves the pressure using the velocity and puts on the grid, debug only, uses eigen to solve on CPU
    static void SolvePressure_EIGEN(MAC2D* gridAnt);
    static void SolvePressure_AMGX(MAC2D* gridAnt);

    //takes the pressure on the grid and crrects the velocity
    static void ProjectPressure(MAC2D* grid);

    static inline Tensor<double,2> GetPressureMask(int i,int j){ return PRESSURE_MASK[i * ((Nx))  + (j)   ];};
    static inline void SetPressureMask(int i,int j,Tensor<double,2> value){PRESSURE_MASK[i * ((Nx))  + (j)   ] = value;};
    static inline int GetIDP(int i,int j){ return IDP(i * ((Nx))  + (j)   );};
    static inline void SetIDP(int i,int j,int value){IDP(i * ((Nx))  + (j)) = value;};

    static int GetSolverIterations();



};

#endif


