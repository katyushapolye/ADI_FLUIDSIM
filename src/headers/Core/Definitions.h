
#ifndef DEFINITIONS_H
#define DEFINITIONS_H


#include "MAC.h"



//empty cell might just as well be outflow
#define FLUID_CELL  0
#define SOLID_CELL  1
#define EMPTY_CELL  2
#define INFLOW_CELL 3



struct SIMULATION_CONFIG{

    double dh;
    double dt;

    double RE;
    double EPS;
    double RHO = 1.0;

    int Nx;
    int Ny;
    int Nz;

    int GRID_SIZE = 16;
    double TOLERANCE = 1E-5;

    bool NEEDS_COMPATIBILITY_CONDITION = false;

    //FLIP
    int PARTICLES_PER_CELL = 4;
    Vec3 f;
    double ALPHA = 1.0;
    bool PHYSICAL_DIFFUSION = true;


    Domain domain;
    LevelConfiguration level;

    Vec3(*VelocityBoundaryFunction)(double, double, double,double);
    double(*PressureBoundaryFunction)(double, double, double,double);
    int(*SolidMaskFunction)(int,int,int);

    Vec2(*VelocityBoundaryFunction2D)(double, double, double);
    double(*PressureBoundaryFunction2D)(double, double, double);
    int(*SolidMaskFunction2D)(int,int);

    
    MAC* GRID_SOL;
    MAC* GRID_ANT;

    std::string ExportPath;
    double CHARACTERISTIC_LENGTH = 0.1;
    double MEAN_VELOCITY = 1.0;

    double lastPressureMatAssemblyTime;
    double lastPressureSolveTime;
    double lastADISolveTime;
    double lastParticleUpdateTime;

    double pressureResidual;
    

};






struct SimulationTelemetry {
    std::vector<double> time;
    std::vector<double> div_sum;
    std::vector<double> div_sum_before_proj;
    std::vector<double> advcfl;
    std::vector<double> diffcfl;
    std::vector<double> residual;
    std::vector<double> cpu_time;
    std::vector<double> gpu_time;
    std::vector<double> particle_time;
    std::vector<double> pressure_residual; 

    size_t max_samples = 5000;

    void Push(double t,
              double div,
              double div_b4,
              double advcfl_,
              double diffcfl_,
              double res,
              double cpu,
              double gpu,
              double particle)
    {
        auto push = [&](std::vector<double>& v, double x) {
            v.push_back(x);
            if (v.size() > max_samples)
                v.erase(v.begin());
        };

        push(time, t);
        push(div_sum, div);
        push(div_sum_before_proj, div_b4);
        push(advcfl, advcfl_);
        push(diffcfl, diffcfl_);
        push(residual, res);
        push(cpu_time, cpu);
        push(gpu_time, gpu);
        push(particle_time, particle);
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
    // Particle export
    bool exportParticleData = false;
    
    std::ofstream telemetryFile;
    bool fileInitialized = false;
};


//Type parameters
enum class SIM_TYPES {
    ADI,
    FLIP,
    SPH,
};

inline std::map<std::string, SIM_TYPES> TYPES = {
    {"ADI", SIM_TYPES::ADI},
    {"FLIP", SIM_TYPES::FLIP},
    {"SPH", SIM_TYPES::SPH}
};
inline SIM_TYPES SIM_TYPE = SIM_TYPES::ADI;

//Performance parameters
inline int DIMENSION = 3;
inline bool GPU_ACCELERATION = 1;
inline int THREAD_COUNT = 4;
inline bool ADAPTATIVE_TIMESTEP = 0;
inline double MAX_CFL = 1.0;


// Simulation parameters
inline SIMULATION_CONFIG SIMULATION;


inline SimulationTelemetry TELEMETRY;
inline AerodynamicInformation AERODYNAMICS;
inline ExportSettings EXPORT_SETTINGS;









#endif


