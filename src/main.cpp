/*Made by Raphael Laroca*/
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <sstream>
#include <amgx_c.h>

#include "headers/MAC.h"
#include "headers/ADI.h"
#include "headers/PressureSolver.h"
#include "headers/Functions.h"
#include "headers/Definitions.h"
#include "headers/ConfigReader.h"


void InitializeSimulation(const std::string& configFile = "simulation_config.txt");
void PrintProgressAndExport(int IT,int F_IT,double endTotal,double startTotal,double b4ProjectDiv,int &frame);


int main(int argc, char *argv[])
{

    omp_set_num_threads(4);


    InitializeSimulation();

   

   

    
    //what needs to be done
    //on projection, only check cells which are faces of fluids fluid or fluid-empty, see if there is an efficient way to do it (pre compute mask) OK!
    //on pressuure, only impose non-homogeneous neumman if it is an entryflow condition - done, kinda of
    //on ADI, check the solid mask for obstacles and throw zero on the diagonal, zero up and down and zero on font 
    //on set newman, onmy set if it is empty and fin the neighboor which is not empty
    // REMBER TO UNDO THESE DO TO THE OTHER EXPERIMENTS!!!
    //        -> laplacian mask now takes into account the empty cell -> DONE! took care of that, inflow, empty, solid and fluid cells
    //        -> pressure correction on last j cell -> DONE! -update mask for each velocity, it checks if it sits on a solid face
    //        -> Neumman imposition on the border velocity for correction purposes on each ADI STEP, ->Maybe we can repurpose the mask??, rely on branch prediction to optimize ifs?



    /*
    PLEASE FOR THE LOVE OF GOD FIND A CLEVVER WAY TO IMPOSE THE NEUMANN BOUNDARIES ON VELOCITY, ALSO IN PRESSURE!
    THE PROJECTION UPDATE ALREADY CHECKS FOR THE FLUID INTERFACE
    CHECK ALSO THE AGRESSIVE INLINING IM DOING, MAYBE IF I TONE IT DOWN THE PERFORMANCE IS BETTER??*/

    
    int IT = 1;
    int frame = 1;
    double time = 0.0;
    double tF = 100000.0;

    int F_IT = tF / SIMULATION.dt;
    double start, end;
    double startTotal, endTotal;
    double b4Project = 0.0;
    std::string filename = "";
    while (time < tF)
    {
        startTotal = GetWallTime();

        // Difusion + Convection
        ADI::SolveADIStep(SIMULATION.GRID_ANT, SIMULATION.GRID_SOL, time);
        // Pressure Projection
        if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) SIMULATION.GRID_SOL->SetNeumannBorder();
        PressureSolver::SolvePressure_AMGX(SIMULATION.GRID_SOL);
        b4Project = SIMULATION.GRID_SOL->GetDivSum();
        PressureSolver::ProjectPressure(SIMULATION.GRID_SOL); 
        time += SIMULATION.dt;


        endTotal = GetWallTime();






        PrintProgressAndExport(IT,F_IT,endTotal,startTotal,b4Project,frame);


        if (SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL) < SIMULATION.TOLERANCE )
        {   if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) SIMULATION.GRID_SOL->SetNeumannBorder();
            SIMULATION.GRID_SOL->ExportGrid(frame);
            frame++;
            break;
        }

        SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);

        IT++;
    }
    



    //cleanup everything later
   



    return 0;
}

void InitializeSimulation(const std::string& configFile) {

    ConfigReader::loadConfig(SIMULATION, configFile);
    SIMULATION.GRID_ANT->ExportGrid(0);
    
    std::cout << "MAC Grid initialized - Parameters\n"
              << "-dt = " << std::to_string(SIMULATION.dt) << "\n"
              << "-dh = " << std::to_string(SIMULATION.dh) 
              << "\n-Nx = " << std::to_string(SIMULATION.Nx) 
              << "\n-Ny = " << std::to_string(SIMULATION.Ny)
              << "\n-Nz = " << std::to_string(SIMULATION.Nz) << std::endl;
    std::cout << "Total node count: " 
              << (SIMULATION.Nx * SIMULATION.Ny * SIMULATION.Nz) * 3 
              << " - Re = " << SIMULATION.RE 
              << " - Grid size: " << SIMULATION.GRID_SIZE << " - Tolerance: " << SIMULATION.TOLERANCE <<std::endl;
    
    ADI::InitializeADI(SIMULATION.GRID_SOL, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction, ZERO, SIMULATION.PressureBoundaryFunction);
    PressureSolver::InitializePressureSolver(SIMULATION.GRID_SOL, SIMULATION.dt);
}

void PrintProgressAndExport(int IT,int F_IT,double endTotal,double startTotal,double b4ProjectDiv,int &frame){
    WriteToCSV(SIMULATION.GRID_SOL->GetDivSum(),LevelConfigurationToString(SIMULATION.level),std::to_string( SIMULATION.GRID_SIZE),"Divergency");
    WriteToCSV(SIMULATION.lastADISolveTime,LevelConfigurationToString(SIMULATION.level),std::to_string( SIMULATION.GRID_SIZE),"TimeADI");
    WriteToCSV(SIMULATION.lastPressureSolveTime,LevelConfigurationToString(SIMULATION.level),std::to_string( SIMULATION.GRID_SIZE),"TimePressure");
    WriteToCSV(PressureSolver::GetSolverIterations(),LevelConfigurationToString(SIMULATION.level),std::to_string( SIMULATION.GRID_SIZE),"ItPressure");
    if(IT%30 == 0){
        if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) SIMULATION.GRID_SOL->SetNeumannBorder();
        SIMULATION.GRID_SOL->ExportGrid(frame);
        frame++;
    }
    printf("====================== ITERATION %d ==========================\n",IT);
    printf("It %d of %d                                \n", IT, F_IT);
    fflush(stdout);
    printf("Pressure Converged in %d it                \n", PressureSolver::GetSolverIterations());
    fflush(stdout);
    printf("Residual = %.10f                           \n", SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL));
    fflush(stdout);
    printf("It Time: %f s                              \n", endTotal - startTotal);
    fflush(stdout);
    printf("divsum (Bfr/Aft)= %.17f / %.17f            \n", b4ProjectDiv, SIMULATION.GRID_SOL->GetDivSum());
    fflush(stdout);
    printf("Adv.CFL: %f                                \n", (SIMULATION.GRID_SOL->GetMaxVelocity()/SIMULATION.dh)*SIMULATION.dt);
    fflush(stdout);
    printf("ADI Avrg Thread CPU Time: %f s             \n", (SIMULATION.lastADISolveTime));
    fflush(stdout);
    printf("Pressure Solve GPU Time: %f s              \n", (SIMULATION.lastPressureSolveTime));
    fflush(stdout);
    //printf("\rIt %d of %d -- Res = %.10f -- It Time: %f s - divsum (Bfr/Aft)= %.17f / %.17f - Adv.CFL: %f", IT, F_IT, SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL), endTotal - startTotal, b4Project,SIMULATION.GRID_SOL->GetDivSum(),(SIMULATION.GRID_SOL->GetMaxVelocity()/SIMULATION.dh)*SIMULATION.dt);
    printf("=============================================================\n",IT);
    fflush(nullptr);
}


