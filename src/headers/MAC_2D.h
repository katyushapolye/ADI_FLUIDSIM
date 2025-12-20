#include "Definitions.h"
#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <filesystem>

#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef MAC2D_H
#define MAC2D_H


class MAC2D
{
public:

    VectorXd u;
    VectorXd v;

    VectorXd p; 


    VectorXd SOLID_MASK;
    
    VectorXd U_UPDATE_MASK;
    VectorXd V_UPDATE_MASK;


public:


    int Nx;
    int Ny;



    Domain2D omega;
    double dh;

    MAC2D();


    void InitializeGrid(Domain2D omega);

    void SetLevelGeometry(int(*SolidMaskFunction)(int,int));

    void SetGrid( Vec2(*VelocityFunction)(double, double, double) , double (*PressureFunction)(double, double, double),double time );

    void SetBorder(Vec2(*VelocityFunction)(double, double, double) , double (*PressureFunction)(double, double, double),double t);

    void SetNeumannBorder();

    void SetNeumannBorderPressure();
    
    double GetDivergencyAt(int i,int j);

    double GetGradPxAt(int i,int j);
    double GetGradPyAt(int i,int j);
    double GetGradPzAt(int i,int j);

    void ExportGrid(int iteration);
    void ExportGridOpenFOAM(int iteration);
    void ExportGridVTK(int iteration);

    double MaxAbsoluteDifference(MAC2D& grid);
    double MaxAbsoluteDifferencePressure(MAC2D& grid);

    double GetMaxVelocity();

    double GetDivSum();

    int GetFluidCellCount();

    //copies the arg to this grid
    void CopyGrid(MAC2D& grid);

    void DestroyGrid();
    




    //interpolation functions
    //gets thhe value of V at node position u_(i,j,k)
    double getVatU(int i,int j);
    double getUatV(int i,int j);



    inline double GetU(int i,int j){ return this->u[i * ((Nx+1))  + (j)   ];};
    inline void SetU(int i,int j,double value){this->u[i * ((Nx+1))  + (j)  ] = value;}

    inline double GetU_Update_Mask(int i,int j){ return this->U_UPDATE_MASK[i * ((Nx+1))  + (j)   ];};
    inline void SetU_Update_Mask(int i,int j,int value){this->U_UPDATE_MASK[i * ((Nx+1))  + (j) ] = value;}

    inline double GetV(int i,int j){ return this->v[i * ((Nx))  + (j)   ];};
    inline void SetV(int i,int j, double value){this->v[i * ((Nx))  + (j)  ] = value;}
    inline double GetV_Update_Mask(int i,int j){ return this->V_UPDATE_MASK[i * ((Nx))  + (j)   ];};
    inline void SetV_Update_Mask(int i,int j,int value){this->V_UPDATE_MASK[i * ((Nx))  + (j)   ] = value;}


    inline double GetP(int i,int j){ return this->p[i * ((this->Nx))  + (j)  ];};
    inline void SetP(int i,int j,double value){this->p[i * ((this->Nx))  + (j)  ] = value;};

    inline double GetSolid(int i,int j){ return this->SOLID_MASK[i * ((this->Nx))  + (j)  ];};
    inline void SetSolid(int i,int j,int value){this->SOLID_MASK(i * ((this->Nx))  + (j)  ) = value;};
};
#endif


