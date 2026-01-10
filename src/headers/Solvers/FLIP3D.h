#include "../Core/MAC.h"
#include "../Utils.h"
#include "../Core/Functions.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>  
#include "math.h"
#include <omp.h>

#ifndef FLIP3D_H
#define FLIP3D_H
// this desperately needs paralelization!
class FLIP3D{
public:
    struct Particle
    {
        bool isActive = false; // just for ease of allocation
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double u = 0.0;
        double v = 0.0;
        double w = 0.0;
    };

    static Particle* particles;
    static int maxParticle;
    static int particleCount;
    static int particlePerCell;
    static double alpha;
    static std::vector<std::vector<std::vector<std::vector<int>>>> SPACE_HASH; // 4D vector for 3D spatial hashing
    static MAC gridAnt;
    static Vec3 queuedAcceleration;

public:
    static void InitializeFLIP(MAC* grid, double dt, double alpha = 1.0);
    static void FLIP_StepBeforeProjection(MAC* grid, double dt);
    static void FLIP_StepAfterProjection(MAC* grid, double dt);
    static void ExportParticles(int IT);
    
    // Wrapper functions for the manager
    static void FLIP_Momentum(MAC* gridAnt, MAC* gridSol, double time);
    static void FLIP_Pressure(MAC* grid);
    static void FLIP_Correction(MAC* grid);

private:
    static double k(double x, double y, double z);
    static double h(double r);
    static void UpdateSpaceHash();
    static void UpdateFluidCells(MAC* grid);
    static void UpdateParticles(double dt);
    static void ReseedParticles(MAC* grid);
    static void ParticleToGrid(MAC* grid);
    static void GridToParticle(MAC* grid);
    
    // This function is only used if another part of the code must add force to the simulation
    static void QueueAcceleration(Vec3 a);
    static double GetTotalKineticEnergy();
    static double GetTotalPotentialEnergy();
};

#endif