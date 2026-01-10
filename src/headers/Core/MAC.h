#include "Definitions.h"
#include "../Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <filesystem>

#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef MAC_H
#define MAC_H


//All (sensible) functions are overloaded to 2D and 3D, those who are not (no parameters) do the check themselves 

class MAC
{
public:

    VectorXd u;
    VectorXd v;
    VectorXd w;
    VectorXd p; 


    VectorXd SOLID_MASK;
    
    VectorXd U_UPDATE_MASK;
    VectorXd V_UPDATE_MASK;
    VectorXd W_UPDATE_MASK;

public:


    int Nx;
    int Ny;
    int Nz;


    Domain omega;
    double dh;
    bool is2D;

    MAC();


    void InitializeGrid(Domain omega,bool is2d = false);

    void SetLevelGeometry(int(*SolidMaskFunction)(int,int,int));

    void SetGrid( Vec3(*VelocityFunction)(double, double, double,double) , double (*PressureFunction)(double, double, double,double),double time );

    void SetBorder(Vec3(*VelocityFunction)(double, double, double,double) , double (*PressureFunction)(double, double, double,double),double t);

    void SetNeumannBorder();

    void SetNeumannBorderPressure();
    
    double GetDivergencyAt(int i,int j,int k);

    double GetGradPxAt(int i,int j,int k);
    double GetGradPyAt(int i,int j,int k);
    double GetGradPzAt(int i,int j,int k);

    void ExportGrid(int iteration);

    void ExportGridVTK(int iteration);

    double MaxAbsoluteDifference(MAC& grid);
    double MaxAbsoluteDifferencePressure(MAC& grid);

    double GetMaxVelocity();
    int GetFluidCellCount();

    double GetDivSum();

    //copies the arg to this grid
    void CopyGrid(MAC& grid);
    void CopySolidMask(MAC& grid);
    void DestroyGrid();

    void AddAcceleration(Vec3 a,double dt);
    void AddAcceleration(Vec2 a,double dt);
    void ResetFluidCells();
    

    //interpolation functions
    //gets thhe value of V at node position u_(i,j,k)
    double getVatU(int i,int j,int k);
    double getWatU(int i,int j,int k);

    double getUatV(int i,int j,int k);
    double getWatV(int i,int j,int k);

    double getUatW(int i,int j,int k);
    double getVatW(int i,int j,int k);


    inline double GetU(int i,int j, int k){ return this->u[i * ((Nx+1)*Nz)  + (j*Nz)  + k ];};
    inline void SetU(int i,int j, int k,double value){this->u[i * ((Nx+1)*Nz)  + (j*Nz)  + k ] = value;}

    inline double GetU_Update_Mask(int i,int j, int k){ return this->U_UPDATE_MASK[i * ((Nx+1)*Nz)  + (j*Nz)  + k ];};
    inline void SetU_Update_Mask(int i,int j, int k,int value){this->U_UPDATE_MASK[i * ((Nx+1)*Nz)  + (j*Nz)  + k ] = value;}

    inline double GetV(int i,int j, int k){ return this->v[i * ((Nx)*Nz)  + (j*Nz)  + k ];};
    inline void SetV(int i,int j, int k,double value){this->v[i * ((Nx)*Nz)  + (j*Nz)  + k ] = value;}
    inline double GetV_Update_Mask(int i,int j, int k){ return this->V_UPDATE_MASK[i * ((Nx)*Nz)  + (j*Nz)  + k ];};
    inline void SetV_Update_Mask(int i,int j, int k,int value){this->V_UPDATE_MASK[i * ((Nx)*Nz)  + (j*Nz)  + k ] = value;}

    inline double GetW(int i,int j, int k){ return this->w[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ];};
    inline void SetW(int i,int j, int k,double value){this->w[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ] = value;};
    inline double GetW_Update_Mask(int i,int j, int k){ return this->W_UPDATE_MASK[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ];};
    inline void SetW_Update_Mask(int i,int j, int k,int value){this->W_UPDATE_MASK[i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ] = value;}

    inline double GetP(int i,int j, int k){ return this->p[i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ];};
    inline void SetP(int i,int j, int k,double value){this->p[i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ] = value;};

    inline double GetSolid(int i,int j, int k){ return this->SOLID_MASK[i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ];};
    inline void SetSolid(int i,int j, int k,int value){this->SOLID_MASK[i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ] = value;};



    //================== 2D OVERLOADS ===============================

    void InitializeGrid(Domain2D omega);


    void SetLevelGeometry(int(*SolidMaskFunction)(int,int));

    void SetGrid( Vec2(*VelocityFunction)(double, double, double) , double (*PressureFunction)(double, double, double),double time );

    void SetBorder(Vec2(*VelocityFunction)(double, double, double) , double (*PressureFunction)(double, double, double),double t);

    
    double GetDivergencyAt(int i,int j);

    double GetGradPxAt(int i,int j);
    double GetGradPyAt(int i,int j);
    double GetGradPzAt(int i,int j);



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


