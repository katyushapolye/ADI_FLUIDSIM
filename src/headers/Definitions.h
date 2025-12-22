
#ifndef DEFINITIONS_H
#define DEFINITIONS_H


#include "MAC.h"
#include "MAC_2D.h"


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
    double CHARACTERISTIC_LENGTH = 0.1;
    double MEAN_VELOCITY = 1.0;

    double lastPressureMatAssemblyTime;
    double lastPressureSolveTime;
    double lastADISolveTime;

    double pressureResidual;
    

};

struct SIMULATION_CONFIG_2D{

    double dh;
    double dt;

    double RE;
    double EPS;

    int Nx;
    int Ny;


    int GRID_SIZE = 16;
    double TOLERANCE = 1E-5;

    bool NEEDS_COMPATIBILITY_CONDITION = false;


    Domain2D domain;
    LevelConfiguration level;

    Vec2(*VelocityBoundaryFunction)(double,  double,double);
    double(*PressureBoundaryFunction)(double, double,double);
    int(*SolidMaskFunction)(int,int);

    
    MAC2D* GRID_SOL;
    MAC2D* GRID_ANT;

    std::string ExportPath;
    double CHARACTERISTIC_LENGTH = 0.1;
    double MEAN_VELOCITY = 1.0;

    double lastPressureMatAssemblyTime;
    double lastPressureSolveTime;
    double lastADISolveTime;

    double pressureResidual;
    

};




struct SimulationTelemetry {
    std::vector<double> time;
    std::vector<double> div_sum;
    std::vector<double> div_sum_before_proj;
    std::vector<double> cfl;
    std::vector<double> residual;
    std::vector<double> cpu_time;
    std::vector<double> gpu_time;
    std::vector<double> pressure_residual; 

    size_t max_samples = 5000;

    void Push(double t,
              double div,
              double div_b4,
              double cfl_,
              double res,
              double cpu,
              double gpu)
    {
        auto push = [&](std::vector<double>& v, double x) {
            v.push_back(x);
            if (v.size() > max_samples)
                v.erase(v.begin());
        };

        push(time, t);
        push(div_sum, div);
        push(div_sum_before_proj, div_b4);
        push(cfl, cfl_);
        push(residual, res);
        push(cpu_time, cpu);
        push(gpu_time, gpu);
    }
};

struct AerodynamicInformation{
    std::vector<double> time;
    std::vector<double> pressure_drop;
    std::vector<double> Cd;
    std::vector<double> Cl;

    std::vector<double> LtoDRatio;
};


struct ExportSettings {
    bool enabled = true;
    int telemetryInterval = 1;  // Export telemetry every N frames
    int gridInterval = 1;      // Export grid every N frames
    int lastExportedTelemetryFrame = 0;
    int lastExportedGridFrame = 0;
    int gridExportCounter = 1;  // Counter for grid file naming
    
    // Telemetry export options
    bool exportTime = true;
    bool exportDivSum = true;
    bool exportDivSumBeforeProj = true;
    bool exportCFL = true;
    bool exportResidual = true;
    bool exportCPUTime = true;
    bool exportGPUTime = true;
    
    // Aerodynamics export options
    bool exportCl = true;
    bool exportCd = true;
    bool exportPressureDrop = true;
    bool exportLtoDRatio = true;
    
    // Grid export
    bool exportGridData = false;
    
    std::ofstream telemetryFile;
    bool fileInitialized = false;
};




inline int DIMENSION = 3;

inline SIMULATION_CONFIG SIMULATION;
inline SIMULATION_CONFIG_2D SIMULATION2D;

inline SimulationTelemetry TELEMETRY;
inline AerodynamicInformation AERODYNAMICS;
inline ExportSettings EXPORT_SETTINGS;


#endif


