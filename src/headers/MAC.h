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

#ifndef MAC_H
#define MAC_H



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

    MAC();


    void InitializeGrid(Domain omega);

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

    double GetDivSum();

    //copies the arg to this grid
    void CopyGrid(MAC& grid);

    void DestroyGrid();
    




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
    inline void SetSolid(int i,int j, int k,int value){this->SOLID_MASK(i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ) = value;};
};
#endif


