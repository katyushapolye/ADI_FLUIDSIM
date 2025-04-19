#ifndef ADI_H
#define ADI_H

#include "MAC.h"
#include "Utils.h"
#include "Functions.h"
#include "Definitions.h"
#include <omp.h>
#include <Eigen/Dense>



using Eigen::MatrixXd;
using Eigen::VectorXd;


class ADI
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

    //Y dir matrixes
    static MatrixXd U_Z_Matrix;
    static MatrixXd V_Z_Matrix;
    static MatrixXd W_Z_Matrix;
    static MatrixXd U_Z_CONV_Matrix;
    static MatrixXd V_Z_CONV_Matrix;
    static MatrixXd W_Z_CONV_Matrix;
    static MatrixXd U_Z_DIFF_Matrix;
    static MatrixXd V_Z_DIFF_Matrix;
    static MatrixXd W_Z_DIFF_Matrix;

    static VectorXd U_Z_Font;
    static VectorXd V_Z_Font;
    static VectorXd W_Z_Font;

    static VectorXd U_Z_SOL;
    static VectorXd V_Z_SOL;
    static VectorXd W_Z_SOL;

    static MAC X_STEP_SOL;
    static MAC Y_STEP_SOL;
    static MAC Z_STEP_SOL;





    static double SIG;
    static double dt;
    static double dh;

    static Vec3(*VelocityBorderFunction)(double, double, double,double);
    static Vec3(*VelocityFontFunction)(double, double, double,double);
    static double(*PressureFunction)(double, double, double,double);

    static void SolveADI_X_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

//OPENMP functions
    static void SolveADI_X_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    
    static void SolveADI_X_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

//Solid aware OPENMP functions, slower but they know of obstacles
    static void SolveADI_X_U_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_U_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_V_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    
    static void SolveADI_X_W_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_W_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_W_Step_OPENMP_SOLID_AWARE(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    
public:
    static void InitializeADI(MAC* grid,double dt,Vec3(*VelocityBorderFunction)(double, double, double,double),Vec3(*VelocityFont)(double, double, double,double),double(*PressureFunction)(double, double, double,double));
    static void SolveADIStep(MAC* gridAnt,MAC* gridSol,double time);

};
#endif


