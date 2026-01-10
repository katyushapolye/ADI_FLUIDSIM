#ifndef SIMULATIONMANAGER_H
#define SIMULATIONMANAGER_H

#include <iostream>
#include "MAC.h"
#include "ConfigReader.h"
#include "../Solvers/ADI3D.h"
#include "../Solvers/ADI2D.h"
#include "../Solvers/PressureSolver3D.h"
#include "../Solvers/PressureSolver2D.h"
#include "../Solvers/FLIP2D.h"
#include "../Solvers/FLIP3D.h"
#include "../Solvers/SPH2D.h"
#include "GridVisualizer.h"

class SimulationManager{

public:
    static void InitializeSimulation(const std::string& configFile = "simulation_config.txt");
    static void InitializeExportTelemetry();
    static void StepSimulation(GridVisualizer* visualizer);

private:
    
    static double currentTime;
    static unsigned int currentIT;
    static unsigned int currentFrame;
    static double lastFrameTime;

    static double finalTime;
    static unsigned int finaIT;



    static bool stepOnce;
    static bool isRunning;

    static double uncorrectedDiv;
    static double correctedDiv;
    static double residual;
    static double diffusionCFL;
    static double advectionCFL;

    //function pointers that gets setup to different solvers
    static void (*momentumStep)(MAC* gridAnt, MAC* gridSol,double time);
    static void (*pressureStep)(MAC* grid);
    static void (*correctionStep)(MAC* grid);

    

    // telemetry update functions
    static void UpdateTelemetry();
    static void UpdateAerodynamicsTelemetry();
    static void UpdateExportTelemetry();
    static void ExportData();

    

};

void (*SimulationManager::momentumStep)(MAC*, MAC*, double) = nullptr;
void (*SimulationManager::pressureStep)(MAC*) = nullptr;
void (*SimulationManager::correctionStep)(MAC*) = nullptr;

double          SimulationManager::currentTime = 0.0;
unsigned int    SimulationManager::currentIT = 0;
unsigned int    SimulationManager::currentFrame = 0;
double          SimulationManager::lastFrameTime = 0.0;
double          SimulationManager::finalTime = 100.0;
unsigned int    SimulationManager::finaIT = 100000;
bool            SimulationManager::stepOnce = false;
bool            SimulationManager::isRunning = false;
double          SimulationManager::uncorrectedDiv = 0.0;
double          SimulationManager::correctedDiv = 0.0;
double          SimulationManager::diffusionCFL = 0.0;
double          SimulationManager::advectionCFL = 0.0;
double          SimulationManager::residual = 0.0;

void SimulationManager::UpdateTelemetry(){
    if(SIM_TYPE != SIM_TYPES::SPH){
        advectionCFL = (SIMULATION.GRID_SOL->GetMaxVelocity() / SIMULATION.dh) * SIMULATION.dt;
        diffusionCFL = (SIMULATION.EPS* SIMULATION.dt) / (SIMULATION.dh*SIMULATION.dh);
    }
    else{
        double maxVel = SPH2D::GetMaxVelocity();
        advectionCFL = (maxVel / SPH2D::h) * SIMULATION.dt;
        diffusionCFL = (SIMULATION.EPS * SIMULATION.dt) / (SPH2D::h * SPH2D::h);
    }

        TELEMETRY.Push(
            currentTime ,
            correctedDiv,
            uncorrectedDiv,
            advectionCFL,
            diffusionCFL,
            residual,
            SIMULATION.lastADISolveTime,
            SIMULATION.lastPressureSolveTime,
            SIMULATION.lastParticleUpdateTime
        );
}


void SimulationManager::UpdateAerodynamicsTelemetry() {
    if(!(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE)) return;

    if(DIMENSION == 2) {

    
    
    /*pressure drop*/
    AERODYNAMICS.time.push_back(currentTime);
    //make the points coordinate be selectable with imgui
    //there is informtion about the max range on SIMULATION.Nx and SIMULATION.Nx

    int maxX = SIMULATION.Nx;
    int maxY = SIMULATION.Ny;
    float dh = SIMULATION.dh; // Get the grid spacing

    static int x1 = SIMULATION.Nx / 4;
    static int y1 = SIMULATION.Ny / 2;
    static int x2 = 3*SIMULATION.Nx / 4 ;
    static int y2 = SIMULATION.Ny / 2;

    double p1 = SIMULATION.GRID_SOL->GetP(y1, x1);
    double p2 = SIMULATION.GRID_SOL->GetP(y2, x2);
    double pdif = p1 - p2;


    /*
    ImGui::Begin("Pressure Difference");

    // Sliders with physical coordinates displayed
    ImGui::SliderInt("X1", &x1, 0, maxX - 1,"%d", ImGuiSliderFlags_None);
    ImGui::SameLine(); ImGui::Text("= %.3f", x1 * dh);

    ImGui::SliderInt("Y1", &y1, 0, maxY - 1, "%d", ImGuiSliderFlags_None);
    ImGui::SameLine(); ImGui::Text("= %.3f", y1 * dh);

    ImGui::Separator();

    ImGui::SliderInt("X2", &x2, 0, maxX - 1, "%d", ImGuiSliderFlags_None);
    ImGui::SameLine(); ImGui::Text("= %.3f", x2 * dh);

    ImGui::SliderInt("Y2", &y2, 0, maxY - 1, "%d", ImGuiSliderFlags_None);
    ImGui::SameLine(); ImGui::Text("= %.3f", y2 * dh);

    // Compute difference (stays the same)
    double p1 = SIMULATION.GRID_SOL->GetP(y1, x1);
    double p2 = SIMULATION.GRID_SOL->GetP(y2, x2);
    double pdif = p1 - p2;

    // Display with physical positions
    ImGui::Separator();
    ImGui::Text("Point 1: (%.3f, %.3f) -> P = %.6f", x1 * dh, y1 * dh, p1);
    ImGui::Text("Point 2: (%.3f, %.3f) -> P = %.6f", x2 * dh, y2 * dh, p2);
    ImGui::Text("Pressure Difference: %.6f", pdif);
    ImGui::End();*/




    double mi = SIMULATION.EPS; // kinematic viscosity
    MAC* grid = SIMULATION.GRID_SOL;

    double dudx = 0.0;
    double dudy = 0.0;
    double dvdx = 0.0;
    double dvdy = 0.0;

    double Fd = 0.0;
    double Fl = 0.0;
    
    // For all non-boundary cell centers
    for(int i = 1; i < SIMULATION.Ny-1; i++) {
        for(int j = 1; j < SIMULATION.Nx-1; j++) {
            // Skip cells that are not solid
            if(grid->GetSolid(i,j) != SOLID_CELL) {
                continue;
            }

            // Process solid cells that have ANY interface with fluid
            // Right face - normal vector (1,0)
            if(grid->GetSolid(i,j+1) == FLUID_CELL) {
                dudx = (grid->GetU(i,j+2) - 0) / dh;
                dvdx = (grid->getVatU(i,j+2) - 0) / dh;
                dudy = (grid->getUatV(i+1,j+1)- (grid->getUatV(i,j+1))) / (dh);

                // Stress components for right face
                double tau_xx = 1.0 * mi * dudx - grid->GetP(i,j+1);
                double tau_yx = mi * (dudy + dvdx);

                // For right face (normal: 1,0):
                Fd += tau_xx * dh;
                Fl += tau_yx * dh;
            }

            // Left face - normal vector (-1,0)
            if(grid->GetSolid(i,j-1) == FLUID_CELL) {
                dudx = (0 - grid->GetU(i,j-1)) / dh;
                dvdx = (0 - grid->getVatU(i,j-1)) / dh;
                dudy = ((grid->getUatV(i+1,j-1))- (grid->getUatV(i,j-1))) / (dh);

                // Stress components for left face
                double tau_xx = 1.0 * mi * dudx - grid->GetP(i,j-1);
                double tau_yx = mi * (dudy + dvdx);

                // For left face (normal: -1,0):
                Fd += -tau_xx * dh;
                Fl += -tau_yx * dh;
            }

            // Top face - normal vector (0,1)
            if(grid->GetSolid(i+1,j) == FLUID_CELL) {
                dvdy = (grid->GetV(i+2,j) - 0) / dh;
                dvdx = ((grid->getVatU(i+1,j+1))- (grid->getVatU(i+1,j))) / (dh);
                dudy = (grid->getUatV(i+2,j) - 0) / dh; 

                // Stress components for top face
                double tau_xy = mi * (dudy + dvdx);
                double tau_yy = 1.0 * mi * dvdy - grid->GetP(i+1,j);

                // For top face (normal: 0,1):
                Fd += tau_xy * dh;
                Fl += tau_yy * dh;
            }

            // Bottom face - normal vector (0,-1) 
            if(grid->GetSolid(i-1,j) == FLUID_CELL) {
                dvdy = (0 - grid->GetV(i-1,j)) / dh;
                dvdx = (grid->getVatU(i-1,j+1)- (grid->getVatU(i-1,j))) / (dh);
                dudy = (0 - grid->getUatV(i-1,j)) / dh;

                // Stress components for bottom face
                double tau_xy = mi * (dudy + dvdx);
                double tau_yy = 1.0 * mi * dvdy - grid->GetP(i-1,j);

                // For bottom face (normal: 0,-1):
                Fd += -tau_xy * dh;
                Fl += -tau_yy * dh;
            }
        }
    }
    double coeff = 2.0 / (pow(SIMULATION.MEAN_VELOCITY,2)* SIMULATION.CHARACTERISTIC_LENGTH);
    //the coefficient is defined as 2/(Umean^2 L) (L is the characteristic lenghth)
    double Cd = coeff*Fd;
    double Cl = coeff*Fl;


    
    // Store in telemetry for visualization
    AERODYNAMICS.Cl.push_back(Cl);
    AERODYNAMICS.Cd.push_back(Cd);
    AERODYNAMICS.pressure_drop.push_back(pdif);
    AERODYNAMICS.LtoDRatio.push_back(Cl/Cd);
    }
    else if(DIMENSION == 3) {
        // 3D implementation
        
        AERODYNAMICS.time.push_back(currentTime);
        
        int maxX = SIMULATION.Nx;
        int maxY = SIMULATION.Ny;
        int maxZ = SIMULATION.Nz;

        static int x1 = 1*SIMULATION.Nx / 4;
        static int y1 = SIMULATION.Ny / 2;
        static int z1 = SIMULATION.Nz / 2;
        static int x2 = 3*SIMULATION.Nx / 4;
        static int y2 = SIMULATION.Ny / 2;
        static int z2 = SIMULATION.Nz / 2;

        double p1 = SIMULATION.GRID_SOL->GetP(z1, y1, x1);
        double p2 = SIMULATION.GRID_SOL->GetP(z2, y2, x2);
        double pdif = p1 - p2;
        /*

        ImGui::Begin("Pressure Difference 3D");

        // Point 1
        ImGui::SliderInt("X1", &x1, 0, maxX - 1);
        ImGui::SliderInt("Y1", &y1, 0, maxY - 1);
        ImGui::SliderInt("Z1", &z1, 0, maxZ - 1);
        ImGui::Separator();
        
        // Point 2
        ImGui::SliderInt("X2", &x2, 0, maxX - 1);
        ImGui::SliderInt("Y2", &y2, 0, maxY - 1);
        ImGui::SliderInt("Z2", &z2, 0, maxZ - 1);

        // Compute pressure difference
        double p1 = SIMULATION.GRID_SOL->GetP(z1, y1, x1);
        double p2 = SIMULATION.GRID_SOL->GetP(z2, y2, x2);
        double pdif = p1 - p2;

        ImGui::Text("P1 = %.6f", p1);
        ImGui::Text("P2 = %.6f", p2);
        ImGui::Text("Difference = %.6f", pdif);

        ImGui::End();
        */

        double dh = SIMULATION.dh;
        double mi = SIMULATION.EPS; // kinematic viscosity
        MAC* grid = SIMULATION.GRID_SOL;

        // Velocity gradients
        double dudx, dudy, dudz;
        double dvdx, dvdy, dvdz;
        double dwdx, dwdy, dwdz;

        double Fd = 0.0; // Drag force (x-direction)
        double Fl = 0.0; // Lift force (y-direction)
        double Fs = 0.0; // Side force (z-direction)
        
        // For all non-boundary cell centers
        for(int k = 1; k < SIMULATION.Nz-1; k++) {
            for(int i = 1; i < SIMULATION.Ny-1; i++) {
                for(int j = 1; j < SIMULATION.Nx-1; j++) {
                    // Skip cells that are not solid
                    if(grid->GetSolid(k,i,j) != SOLID_CELL) {
                        continue;
                    }

                    double dA = dh * dh; // Face area

                    // ===== RIGHT FACE (X+) - normal vector (1,0,0) =====
                    if(grid->GetSolid(k,i,j+1) == FLUID_CELL) {
                        dudx = (grid->GetU(k,i,j+2) - 0) / dh;
                        dvdx = (grid->getVatU(k,i,j+2) - 0) / dh;
                        dwdx = (grid->getWatU(k,i,j+2) - 0) / dh;
                        dudy = (grid->getUatV(k,i+1,j+1) - grid->getUatV(k,i,j+1)) / dh;
                        dudz = (grid->getUatW(k+1,i,j+1) - grid->getUatW(k,i,j+1)) / dh;

                        // Stress tensor components for right face
                        double tau_xx = 2.0 * mi * dudx - grid->GetP(k,i,j+1);
                        double tau_yx = mi * (dudy + dvdx);
                        double tau_zx = mi * (dudz + dwdx);

                        // Force contribution (normal: 1,0,0)
                        Fd += tau_xx * dA;
                        Fl += tau_yx * dA;
                        Fs += tau_zx * dA;
                    }

                    // ===== LEFT FACE (X-) - normal vector (-1,0,0) =====
                    if(grid->GetSolid(k,i,j-1) == FLUID_CELL) {
                        dudx = (0 - grid->GetU(k,i,j-1)) / dh;
                        dvdx = (0 - grid->getVatU(k,i,j-1)) / dh;
                        dwdx = (0 - grid->getWatU(k,i,j-1)) / dh;
                        dudy = (grid->getUatV(k,i+1,j-1) - grid->getUatV(k,i,j-1)) / dh;
                        dudz = (grid->getUatW(k+1,i,j-1) - grid->getUatW(k,i,j-1)) / dh;

                        // Stress tensor components for left face
                        double tau_xx = 2.0 * mi * dudx - grid->GetP(k,i,j-1);
                        double tau_yx = mi * (dudy + dvdx);
                        double tau_zx = mi * (dudz + dwdx);

                        // Force contribution (normal: -1,0,0)
                        Fd += -tau_xx * dA;
                        Fl += -tau_yx * dA;
                        Fs += -tau_zx * dA;
                    }

                    // ===== TOP FACE (Y+) - normal vector (0,1,0) =====
                    if(grid->GetSolid(k,i+1,j) == FLUID_CELL) {
                        dvdy = (grid->GetV(k,i+2,j) - 0) / dh;
                        dvdx = (grid->getVatU(k,i+1,j+1) - grid->getVatU(k,i+1,j)) / dh;
                        dvdz = (grid->getVatW(k+1,i+1,j) - grid->getVatW(k,i+1,j)) / dh;
                        dudy = (grid->getUatV(k,i+2,j) - 0) / dh;
                        dwdy = (grid->getWatV(k,i+2,j) - 0) / dh;

                        // Stress tensor components for top face
                        double tau_xy = mi * (dudy + dvdx);
                        double tau_yy = 2.0 * mi * dvdy - grid->GetP(k,i+1,j);
                        double tau_zy = mi * (dvdz + dwdy);

                        // Force contribution (normal: 0,1,0)
                        Fd += tau_xy * dA;
                        Fl += tau_yy * dA;
                        Fs += tau_zy * dA;
                    }

                    // ===== BOTTOM FACE (Y-) - normal vector (0,-1,0) =====
                    if(grid->GetSolid(k,i-1,j) == FLUID_CELL) {
                        dvdy = (0 - grid->GetV(k,i-1,j)) / dh;
                        dvdx = (grid->getVatU(k,i-1,j+1) - grid->getVatU(k,i-1,j)) / dh;
                        dvdz = (grid->getVatW(k+1,i-1,j) - grid->getVatW(k,i-1,j)) / dh;
                        dudy = (0 - grid->getUatV(k,i-1,j)) / dh;
                        dwdy = (0 - grid->getWatV(k,i-1,j)) / dh;

                        // Stress tensor components for bottom face
                        double tau_xy = mi * (dudy + dvdx);
                        double tau_yy = 2.0 * mi * dvdy - grid->GetP(k,i-1,j);
                        double tau_zy = mi * (dvdz + dwdy);

                        // Force contribution (normal: 0,-1,0)
                        Fd += -tau_xy * dA;
                        Fl += -tau_yy * dA;
                        Fs += -tau_zy * dA;
                    }

                    // ===== FRONT FACE (Z+) - normal vector (0,0,1) =====
                    if(grid->GetSolid(k+1,i,j) == FLUID_CELL) {
                        dwdz = (grid->GetW(k+2,i,j) - 0) / dh;
                        dwdx = (grid->getWatU(k+1,i,j+1) - grid->getWatU(k+1,i,j)) / dh;
                        dwdy = (grid->getWatV(k+1,i+1,j) - grid->getWatV(k+1,i,j)) / dh;
                        dudz = (grid->getUatW(k+2,i,j) - 0) / dh;
                        dvdz = (grid->getVatW(k+2,i,j) - 0) / dh;

                        // Stress tensor components for front face
                        double tau_xz = mi * (dudz + dwdx);
                        double tau_yz = mi * (dvdz + dwdy);
                        double tau_zz = 2.0 * mi * dwdz - grid->GetP(k+1,i,j);

                        // Force contribution (normal: 0,0,1)
                        Fd += tau_xz * dA;
                        Fl += tau_yz * dA;
                        Fs += tau_zz * dA;
                    }

                    // ===== BACK FACE (Z-) - normal vector (0,0,-1) =====
                    if(grid->GetSolid(k-1,i,j) == FLUID_CELL) {
                        dwdz = (0 - grid->GetW(k-1,i,j)) / dh;
                        dwdx = (grid->getWatU(k-1,i,j+1) - grid->getWatU(k-1,i,j)) / dh;
                        dwdy = (grid->getWatV(k-1,i+1,j) - grid->getWatV(k-1,i,j)) / dh;
                        dudz = (0 - grid->getUatW(k-1,i,j)) / dh;
                        dvdz = (0 - grid->getVatW(k-1,i,j)) / dh;

                        // Stress tensor components for back face
                        double tau_xz = mi * (dudz + dwdx);
                        double tau_yz = mi * (dvdz + dwdy);
                        double tau_zz = 2.0 * mi * dwdz - grid->GetP(k-1,i,j);

                        // Force contribution (normal: 0,0,-1)
                        Fd += -tau_xz * dA;
                        Fl += -tau_yz * dA;
                        Fs += -tau_zz * dA;
                    }
                }
            }
        }



        double coeff = 2.0 / (pow(SIMULATION.MEAN_VELOCITY,2)* SIMULATION.CHARACTERISTIC_LENGTH);
        
        double Cd = coeff * Fd;
        double Cl = coeff * Fl;
        double Cs = coeff * Fs; // Side force coefficient
        
        // Store in telemetry
        AERODYNAMICS.Cl.push_back(Cl);
        AERODYNAMICS.Cd.push_back(Cd);
        AERODYNAMICS.pressure_drop.push_back(pdif);
        AERODYNAMICS.LtoDRatio.push_back(Cl/Cd);
        

    }
}

void SimulationManager::UpdateExportTelemetry() {
    if (!EXPORT_SETTINGS.enabled || !EXPORT_SETTINGS.fileInitialized) return;
    
    EXPORT_SETTINGS.telemetryFile << currentIT;
    
    // Export simulation telemetry
    if (EXPORT_SETTINGS.exportTime) {
        if (!TELEMETRY.time.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.time.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportDivSum) {
        if (!TELEMETRY.div_sum.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.div_sum.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportDivSumBeforeProj) {
        if (!TELEMETRY.div_sum_before_proj.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.div_sum_before_proj.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportCFL) {
        if (!TELEMETRY.advcfl.empty()){
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.advcfl.back();
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.diffcfl.back();
        }
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportResidual) {
        if (!TELEMETRY.residual.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.residual.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportCPUTime) {
        if (!TELEMETRY.cpu_time.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.cpu_time.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    if (EXPORT_SETTINGS.exportGPUTime) {
        if (!TELEMETRY.gpu_time.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.gpu_time.back();
        else
            EXPORT_SETTINGS.telemetryFile << ",0.0";
    }
    
    // Export aerodynamics telemetry (only for 2D with obstacle)
    if ( SIMULATION.level == LevelConfiguration::OBSTACLE || SIMULATION.level == LevelConfiguration::OBSTACLE) {
        if (EXPORT_SETTINGS.exportCl) {
            if (!AERODYNAMICS.Cl.empty())
                EXPORT_SETTINGS.telemetryFile << "," << AERODYNAMICS.Cl.back();
            else
                EXPORT_SETTINGS.telemetryFile << ",0.0";
        }
        
        if (EXPORT_SETTINGS.exportCd) {
            if (!AERODYNAMICS.Cd.empty())
                EXPORT_SETTINGS.telemetryFile << "," << AERODYNAMICS.Cd.back();
            else
                EXPORT_SETTINGS.telemetryFile << ",0.0";
        }
        
        if (EXPORT_SETTINGS.exportPressureDrop) {
            if (!AERODYNAMICS.pressure_drop.empty())
                EXPORT_SETTINGS.telemetryFile << "," << AERODYNAMICS.pressure_drop.back();
            else
                EXPORT_SETTINGS.telemetryFile << ",0.0";
        }
        
        if (EXPORT_SETTINGS.exportLtoDRatio) {
            if (!AERODYNAMICS.LtoDRatio.empty())
                EXPORT_SETTINGS.telemetryFile << "," << AERODYNAMICS.LtoDRatio.back();
            else
                EXPORT_SETTINGS.telemetryFile << ",0.0";
        }
    }
    
    EXPORT_SETTINGS.telemetryFile << "\n";
    EXPORT_SETTINGS.telemetryFile.flush(); // Ensure data is written
}

void SimulationManager::ExportData() {
    if (!EXPORT_SETTINGS.enabled) return;
    


    // Export telemetry at telemetry interval
    if ((currentIT - 1) % EXPORT_SETTINGS.telemetryInterval == 0) {
        UpdateExportTelemetry();
        EXPORT_SETTINGS.lastExportedTelemetryFrame = currentIT;
    }
    
    // Export grid at grid interval (if enabled)
    if (EXPORT_SETTINGS.exportGridData && (currentIT - 1) % EXPORT_SETTINGS.gridInterval == 0) {

        SIMULATION.GRID_SOL->ExportGrid(EXPORT_SETTINGS.gridExportCounter);

    }
    
    if(EXPORT_SETTINGS.exportParticleData && (currentIT - 1) % EXPORT_SETTINGS.gridInterval == 0){
        if(SIM_TYPE == SIM_TYPES::FLIP){
            if(DIMENSION == 2 ) FLIP2D::ExportParticles(EXPORT_SETTINGS.gridExportCounter);
            else FLIP3D::ExportParticles(EXPORT_SETTINGS.gridExportCounter);
        }
        else if(SIM_TYPE == SIM_TYPES::SPH){
             if(DIMENSION == 2 ) SPH2D::ExportParticles(EXPORT_SETTINGS.gridExportCounter);
        }


    }

    //increaasing the counter

    if(EXPORT_SETTINGS.exportGridData || EXPORT_SETTINGS.exportParticleData){
        EXPORT_SETTINGS.lastExportedGridFrame = currentIT;
        EXPORT_SETTINGS.gridExportCounter++;
    }

    
}

void  SimulationManager::InitializeExportTelemetry() {
    if (EXPORT_SETTINGS.fileInitialized) return;
    
    std::string levelStr;
    std::string exportBasePath = "Exports";
    std::string exportPath;
    
    if (DIMENSION == 3) {
        levelStr = LevelConfigurationToString(SIMULATION.level);
        exportPath = exportBasePath + "/" + levelStr + "/" +
            std::to_string(SIMULATION.GRID_SIZE) + "_3D" + "_re" +
            std::to_string(int(SIMULATION.RE)) + "/";
    } else {
        levelStr = LevelConfigurationToString(SIMULATION.level);
        exportPath = exportBasePath + "/" + levelStr + "/" +
            std::to_string(SIMULATION.GRID_SIZE) + "_2D" + "_re" +
            std::to_string(int(SIMULATION.RE)) + "/";
    }
    
    std::filesystem::create_directories(exportPath);
    
    std::string filename = exportPath + "telemetry.csv";
    EXPORT_SETTINGS.telemetryFile.open(filename);
    
    if (!EXPORT_SETTINGS.telemetryFile.is_open()) {
        std::cerr << "Failed to open telemetry export file: " << filename << std::endl;
        return;
    }
    
    // Write CSV header
    EXPORT_SETTINGS.telemetryFile << "iteration";
    
    if (EXPORT_SETTINGS.exportTime) 
        EXPORT_SETTINGS.telemetryFile << ",time";
    if (EXPORT_SETTINGS.exportDivSum) 
        EXPORT_SETTINGS.telemetryFile << ",div_sum";
    if (EXPORT_SETTINGS.exportDivSumBeforeProj) 
        EXPORT_SETTINGS.telemetryFile << ",div_sum_before_proj";
    if (EXPORT_SETTINGS.exportCFL) 
        EXPORT_SETTINGS.telemetryFile << ",advcfl,diffcfl";
    if (EXPORT_SETTINGS.exportResidual) 
        EXPORT_SETTINGS.telemetryFile << ",residual";
    if (EXPORT_SETTINGS.exportCPUTime) 
        EXPORT_SETTINGS.telemetryFile << ",cpu_time";
    if (EXPORT_SETTINGS.exportGPUTime) 
        EXPORT_SETTINGS.telemetryFile << ",gpu_time";
    
    // Aerodynamics headers (only for 2D with obstacle)
    if (SIMULATION.level == LevelConfiguration::OBSTACLE) {
        if (EXPORT_SETTINGS.exportCl) 
            EXPORT_SETTINGS.telemetryFile << ",cl";
        if (EXPORT_SETTINGS.exportCd) 
            EXPORT_SETTINGS.telemetryFile << ",cd";
        if (EXPORT_SETTINGS.exportPressureDrop) 
            EXPORT_SETTINGS.telemetryFile << ",pressure_drop";
        if (EXPORT_SETTINGS.exportLtoDRatio) 
            EXPORT_SETTINGS.telemetryFile << ",ld_ratio";
    }
    
    EXPORT_SETTINGS.telemetryFile << "\n";
    EXPORT_SETTINGS.fileInitialized = true;
}





void SimulationManager::InitializeSimulation(const std::string& configFile){
    ConfigReader::loadConfig(SIMULATION, configFile);
    omp_set_num_threads(THREAD_COUNT);
    Eigen::initParallel();
    TELEMETRY.Push(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0);
    
    std::cout << "MAC Grid initialized - Parameters\n"
              << "-dt = " << std::to_string(SIMULATION.dt) << "\n"
              << "-dh = " << std::to_string(SIMULATION.dh) 
              << "\n-Nx = " << std::to_string(SIMULATION.Nx) 
              << "\n-Ny = " << std::to_string(SIMULATION.Ny)
              << "\n-Nz = " << std::to_string(SIMULATION.Nz) << std::endl;
    std::cout << "Total node count: " 
              << ((SIMULATION.Nx+1) * (SIMULATION.Ny+1) * (SIMULATION.Nz+1)) * 3 
              << " - Re = " << SIMULATION.RE 
              << " - Grid size: " << SIMULATION.GRID_SIZE 
              << " - Tolerance: " << SIMULATION.TOLERANCE << std::endl;

    if(DIMENSION == 3 && SIM_TYPE == SIM_TYPES::ADI){

        ADI3D::InitializeADI(SIMULATION.GRID_SOL, SIMULATION.dt, 
                          SIMULATION.VelocityBoundaryFunction, ZERO, 
                          SIMULATION.PressureBoundaryFunction);
        PressureSolver3D::InitializePressureSolver(SIMULATION.GRID_SOL, false);
        SIMULATION.GRID_ANT->SetBorder(SIMULATION.VelocityBoundaryFunction,SIMULATION.PressureBoundaryFunction,0);

        momentumStep = ADI3D::SolveADIStep;
        if(GPU_ACCELERATION){
            pressureStep = PressureSolver3D::SolvePressure_AMGX;

        }
        else{
              pressureStep = PressureSolver3D::SolvePressure_EIGEN;
        }
        correctionStep = PressureSolver3D::ProjectPressure;
    }
    else if(DIMENSION == 2 && SIM_TYPE == SIM_TYPES::ADI){

        ADI2D::InitializeADI2D(SIMULATION.GRID_SOL, SIMULATION.dt, 
                               SIMULATION.VelocityBoundaryFunction2D, ZERO2D, 
                               SIMULATION.PressureBoundaryFunction2D);
        PressureSolver2D::InitializePressureSolver(SIMULATION.GRID_SOL,false); //do not rebuild the matrix at every iteration
        SIMULATION.GRID_ANT->SetBorder(SIMULATION.VelocityBoundaryFunction2D,SIMULATION.PressureBoundaryFunction2D,0);

        momentumStep = ADI2D::SolveADIStep;
        if(GPU_ACCELERATION){
            pressureStep = PressureSolver2D::SolvePressure_AMGX;

        }
        else{
              pressureStep = PressureSolver2D::SolvePressure_EIGEN;
        }

        correctionStep = PressureSolver2D::ProjectPressure;
    }
    else if(DIMENSION == 2 && SIM_TYPE == SIM_TYPES::FLIP){

        ADI2D::InitializeADI2D(SIMULATION.GRID_SOL, SIMULATION.dt, 
                               SIMULATION.VelocityBoundaryFunction2D, ZERO2D, 
                               SIMULATION.PressureBoundaryFunction2D);
        PressureSolver2D::InitializePressureSolver(SIMULATION.GRID_SOL,true);
        FLIP2D::InitializeFLIP(SIMULATION.GRID_SOL,SIMULATION.dt,SIMULATION.ALPHA); //MAAKE AAN ALPHA LATER ON SIMCONFIG
        SIMULATION.GRID_ANT->SetBorder(SIMULATION.VelocityBoundaryFunction2D,SIMULATION.PressureBoundaryFunction2D,0);

        momentumStep = FLIP2D::FLIP_Momentum;
        if(GPU_ACCELERATION){
            //std::cout << "ERROR - AMGX IS MAYBE BUGGED! - USE AT YOUR OWN RISK!" << std::endl;;
            pressureStep = PressureSolver2D::SolvePressure_AMGX_VARIABLE;

        }
        else{
              pressureStep = PressureSolver2D::SolvePressure_EIGEN;
        }

        correctionStep = FLIP2D::FLIP_Correction;
        
    }
    else if(DIMENSION == 3 && SIM_TYPE == SIM_TYPES::FLIP){

        ADI3D::InitializeADI(SIMULATION.GRID_SOL, SIMULATION.dt, 
                               SIMULATION.VelocityBoundaryFunction, ZERO, 
                               SIMULATION.PressureBoundaryFunction);
        PressureSolver3D::InitializePressureSolver(SIMULATION.GRID_SOL,true);
        FLIP3D::InitializeFLIP(SIMULATION.GRID_SOL,SIMULATION.dt,SIMULATION.ALPHA); //MAAKE AAN ALPHA LATER ON SIMCONFIG
        SIMULATION.GRID_ANT->SetBorder(SIMULATION.VelocityBoundaryFunction,SIMULATION.PressureBoundaryFunction,0);

        momentumStep = FLIP3D::FLIP_Momentum;
        if(GPU_ACCELERATION){
            std::cout << "3D PROPER SOLVE NOT YET IMPLEMENTED!" << std::endl;;
            pressureStep = PressureSolver3D::SolvePressure_AMGX_VARIABLE;

        }
        else{
              pressureStep = PressureSolver3D::SolvePressure_EIGEN;
        }

        correctionStep = FLIP3D::FLIP_Correction;

    }
    else if(DIMENSION == 2 && SIM_TYPE == SIM_TYPES::SPH){
        SPH2D::InitializeSPH2D(1000000);
        momentumStep = SPH2D::AdvanceSPH;
        pressureStep = SPH2D::DummySPH;
        correctionStep = SPH2D::DummySPH;
    }


}


void SimulationManager::StepSimulation(GridVisualizer* visualizer){

    visualizer->UpdateGrid(SIMULATION.GRID_SOL);
    visualizer->Render(currentIT,currentTime,residual,lastFrameTime, isRunning,stepOnce);

    
    if(DIMENSION == 2) SIMULATION.GRID_SOL->SetGrid(ZERO2D,ZERO2D_SCALAR,0); //set everything to zero, this is very important when the grid changes
    else SIMULATION.GRID_SOL->SetGrid(ZERO,ZERO_SCALAR,0); //we might as well make a reset function
    
    if(ADAPTATIVE_TIMESTEP){
        if(SIM_TYPE != SIM_TYPES::SPH){
            advectionCFL = (SIMULATION.GRID_ANT->GetMaxVelocity() / SIMULATION.dh) * SIMULATION.dt;
            diffusionCFL = (SIMULATION.EPS * SIMULATION.dt) / (SIMULATION.dh * SIMULATION.dh);
            double currentCFL = std::max(diffusionCFL, advectionCFL);
            double dt_advection = (MAX_CFL * SIMULATION.dh) / SIMULATION.GRID_ANT->GetMaxVelocity();
            double dt_diffusion = (MAX_CFL * SIMULATION.dh * SIMULATION.dh) / SIMULATION.EPS;
            SIMULATION.dt = std::min(dt_advection, dt_diffusion);
        }
    else{
        double maxVel = SPH2D::GetMaxVelocity();
        double maxForce = SPH2D::GetMaxForce();

        double dt_cfl = MAX_CFL * SPH2D::h / std::max(maxVel, 0.01);
        double dt_force = 0.25 * sqrt(SPH2D::h * SPH2D::mass / std::max(maxForce, 1e-3));
        double dt_viscous = MAX_CFL * SPH2D::h * SPH2D::h / (SIMULATION.EPS + 1e-6);

        // Cap maximum timestep for safety
        double dt_max = 0.01; 

        SIMULATION.dt = std::min({dt_cfl, dt_force, dt_viscous, dt_max});

    }
   
    }

    if (isRunning || stepOnce){
        double start = GetWallTime();

        momentumStep(SIMULATION.GRID_ANT,SIMULATION.GRID_SOL,currentTime);

        if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE){
            SIMULATION.GRID_SOL->SetNeumannBorder(); //this will be refactored when i think of a good way to do this update checking
        } 


        pressureStep(SIMULATION.GRID_SOL);
        uncorrectedDiv = SIMULATION.GRID_SOL->GetDivSum();
        correctionStep(SIMULATION.GRID_SOL);
        
        residual = SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL);
        correctedDiv = SIMULATION.GRID_SOL->GetDivSum();




        UpdateTelemetry();
        UpdateAerodynamicsTelemetry();
        UpdateExportTelemetry();






        SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);

        currentTime += SIMULATION.dt;
        currentIT++;
        stepOnce = false;

        ExportData();

        double end  = GetWallTime();
        lastFrameTime = end-start;
    }
        



}




#endif