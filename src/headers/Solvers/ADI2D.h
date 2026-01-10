#ifndef ADI2D_H
#define ADI2D_H

#include "../Core/MAC.h"
#include "../Core/Functions.h"
#include "../Core/Definitions.h"
#include "Utils.h"

#include <omp.h>
#include <Eigen/Dense>



using Eigen::MatrixXd;
using Eigen::VectorXd;


class ADI2D
{
private:
    //for efficiency, we allocate LOTS of matrices only once
    //X dir matrixes
    static MatrixXd U_X_Matrix;
    static MatrixXd V_X_Matrix;
    static MatrixXd W_X_Matrix;
    static MatrixXd U_X_CONV_Matrix;
    static MatrixXd V_X_CONV_Matrix;
    static MatrixXd W_X_CONV_Matrix;
    static MatrixXd U_X_DIFF_Matrix;
    static MatrixXd V_X_DIFF_Matrix;
    static MatrixXd W_X_DIFF_Matrix;

    static VectorXd U_X_Font;
    static VectorXd V_X_Font;
    static VectorXd W_X_Font;

    static VectorXd U_X_SOL;
    static VectorXd V_X_SOL;
    static VectorXd W_X_SOL;

    //Y dir matrixes
    static MatrixXd U_Y_Matrix;
    static MatrixXd V_Y_Matrix;
    static MatrixXd W_Y_Matrix;
    static MatrixXd U_Y_CONV_Matrix;
    static MatrixXd V_Y_CONV_Matrix;
    static MatrixXd W_Y_CONV_Matrix;
    static MatrixXd U_Y_DIFF_Matrix;
    static MatrixXd V_Y_DIFF_Matrix;
    static MatrixXd W_Y_DIFF_Matrix;

    static VectorXd U_Y_Font;
    static VectorXd V_Y_Font;
    static VectorXd W_Y_Font;

    static VectorXd U_Y_SOL;
    static VectorXd V_Y_SOL;
    static VectorXd W_Y_SOL;


    static MAC X_STEP_SOL;
    static MAC Y_STEP_SOL;





    static double SIG;
    static double dh;

    static Vec2(*VelocityBorderFunction)(double, double, double);
    static Vec2(*VelocityFontFunction)(double, double, double);
    static double(*PressureFunction)(double, double, double);





//OPENMP functions
    static void SolveADI_X_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_U_DIFF_Step_OPENMP(MAC* gridAnt,MAC* gridSol,double time);
    static void SolveADI_Y_U_DIFF_Step_OPENMP(MAC* gridAnt,MAC* gridSol,double time);

    static void SolveADI_X_V_DIFF_Step_OPENMP(MAC* gridAnt,MAC* gridSol,double time);
    static void SolveADI_Y_V_DIFF_Step_OPENMP(MAC* gridAnt,MAC* gridSol,double time);


public:
    static void InitializeADI2D(MAC* grid,double dt,Vec2(*VelocityBorderFunction)(double, double, double),Vec2(*VelocityFont)(double, double, double),double(*PressureFunction)(double, double, double));
    static void SolveADIStep(MAC* gridAnt,MAC* gridSol,double time);
    static void SolveADIDiffusionStep(MAC* gridAnt,MAC* gridSol,double time);

};
#endif


