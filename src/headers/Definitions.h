
#pragma once


#include "MAC.h"


#define FLUID_CELL  0
#define SOLID_CELL  1
#define EMPTY_CELL  2
#define INFLOW_CELL 3

//#define RE 389.0

//add function poointer to
//solid funcion
//etc



struct SIMULATION_CONFIG{
    double dh;
    double dt;

    double RE;
    double EPS;

    int Nx;
    int Ny;
    int Nz;

    int GRID_SIZE = 16;
    double TOLERANCE = 1E-5;

    bool NEEDS_COMPATIBILITY_CONDITION = false;


    Domain domain;
    LevelConfiguration level;

    Vec3(*VelocityBoundaryFunction)(double, double, double,double);
    double(*PressureBoundaryFunction)(double, double, double,double);
    int(*SolidMaskFunction)(int,int,int);

    
    MAC* GRID_SOL;
    MAC* GRID_ANT;

    std::string ExportPath;

    double lastPressureMatAssemblyTime;
    double lastPressureSolveTime;
    double lastADISolveTime;
    

};


inline SIMULATION_CONFIG SIMULATION;

