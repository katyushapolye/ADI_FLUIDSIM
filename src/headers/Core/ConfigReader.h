#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <iostream>

#include "Functions.h"
#include "Definitions.h"
#include "Utils.h"



namespace ConfigReader {
    static std::map<std::string, std::string> parameters;


    static bool loadFromFile(const std::string& filename) {
        std::ifstream configFile(filename);
        if (!configFile.is_open()) {
            std::cerr << "Error: Could not open configuration file " << filename << std::endl;
            return false;
        }

        parameters.clear();
        std::string line;
        while (std::getline(configFile, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '/') {
                continue;
            }

            std::istringstream lineStream(line);
            std::string key, value;
            
            // Format expected: key = value
            if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                parameters[key] = value;
            }
        }
        
        configFile.close();
        return true;
    }


    static std::string getString(const std::string& key, const std::string& defaultValue = "") {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            return it->second;
        }
        return defaultValue;
    }
    

    static double getDouble(const std::string& key, double defaultValue = 0.0) {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            try {
                return std::stod(it->second);
            } catch (const std::exception&) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    

    static int getInt(const std::string& key, int defaultValue = 0) {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            try {
                return std::stoi(it->second);
            } catch (const std::exception&) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    
    // Load configuration into an existing SIMULATION_CONFIG struct
    static void loadConfig(SIMULATION_CONFIG& config, const std::string& filename) {
        if (!loadFromFile(filename)) {
            std::cerr << "Warning: Using default simulation parameters." << std::endl;
            return;
        }
        DIMENSION = getInt("DIMENSION", 3);
        GPU_ACCELERATION = getInt("GPU_ACCELERATION",0);
        THREAD_COUNT = getInt("THREAD_COUNT",4);
        SIM_TYPE = TYPES[getString("TYPE","ADI")];


        ADAPTATIVE_TIMESTEP = getInt("ADAPTATIVE_TIMESTEP",0);
        MAX_CFL =  getDouble("MAX_CFL",1.0);

        SIMULATION.GRID_SOL = new MAC();
        SIMULATION.GRID_ANT = new MAC();
        SIMULATION.domain = Domain();
        
        SIMULATION.domain.x0 = getDouble("DOMAIN_X0", config.domain.x0);
        SIMULATION.domain.xf = getDouble("DOMAIN_XF", config.domain.xf);
        SIMULATION.domain.y0 = getDouble("DOMAIN_Y0", config.domain.y0);
        SIMULATION.domain.yf = getDouble("DOMAIN_YF", config.domain.yf);
        SIMULATION.domain.z0 = getDouble("DOMAIN_Z0", config.domain.z0);
        SIMULATION.domain.zf = getDouble("DOMAIN_ZF", config.domain.zf);
        
        SIMULATION.dt = getDouble("TIME_STEP", config.dt);
        SIMULATION.RE = getDouble("REYNOLDS_NUMBER", config.RE);
        SIMULATION.EPS = getDouble("VISCOSITY", 0.01);
        SIMULATION.GRID_SIZE = getInt("GRID_SIZE", config.GRID_SIZE);
        SIMULATION.TOLERANCE = getDouble("TOLERANCE", config.TOLERANCE);
        SIMULATION.ExportPath = getString("EXPORT_BASE_PATH", "Exports");
        SIMULATION.CHARACTERISTIC_LENGTH = getDouble("CHARACTERISTIC_LENGTH", config.CHARACTERISTIC_LENGTH);
        SIMULATION.MEAN_VELOCITY = getDouble("MEAN_VELOCITY", config.MEAN_VELOCITY);
        SIMULATION.NEEDS_COMPATIBILITY_CONDITION = getInt("NEEDS_COMPATIBILITY_CONDITION",config.NEEDS_COMPATIBILITY_CONDITION);
        
        
        SIMULATION.GRID_SOL->InitializeGrid(SIMULATION.domain,DIMENSION == 3? false:true);
        SIMULATION.GRID_ANT->InitializeGrid(SIMULATION.domain,DIMENSION == 3? false:true);
        
        SIMULATION.dh = (SIMULATION.GRID_ANT->dh);
        SIMULATION.Nx = SIMULATION.GRID_ANT->Nx;
        SIMULATION.Ny = SIMULATION.GRID_ANT->Ny;
        SIMULATION.Nz = SIMULATION.GRID_ANT->Nz;
        
        std::string levelType = getString("LEVEL_TYPE", "STEP");
        if (levelType == "STEP") {
            SIMULATION.level = LevelConfiguration::STEP;
        } else if (levelType == "CAVITY") {
            SIMULATION.level = LevelConfiguration::LID_CAVITY;
        } 
        else if (levelType == "OBSTACLE"){
            SIMULATION.level = LevelConfiguration::OBSTACLE;
        }
        else if (levelType == "DAMBREAK"){
            SIMULATION.level = LevelConfiguration::DAMBREAK;
        }
        else {
            std::cerr << "Warning: Unknown level type '" << levelType << "'. Using default." << std::endl;
        }

        //flip stuff
        SIMULATION.PHYSICAL_DIFFUSION = getInt("PHYSICAL_DIFFUSION",1);
        SIMULATION.PARTICLES_PER_CELL = getInt("PARTICLES_PER_CELL");
        SIMULATION.ALPHA = getDouble("ALPHA",1.0);
        SIMULATION.f.u = getDouble("Fu",0.0);
        SIMULATION.f.v = getDouble("Fv",0.0);
        SIMULATION.f.w = getDouble("Fw",0.0);


        
        if(DIMENSION == 3 && SIM_TYPE == SIM_TYPES::ADI){
            

            // Level geometry, boundary functions.
            if (SIMULATION.level == LevelConfiguration::STEP) {
                SIMULATION.SolidMaskFunction = BACKWARDS_FACING_STEP_SOLID_MASK;
                SIMULATION.VelocityBoundaryFunction = BACKWARDS_FACING_STEP;
                SIMULATION.PressureBoundaryFunction = BACKWARDS_FACING_STEP_PRESSURE;
            }
            else if (SIMULATION.level == LevelConfiguration::LID_CAVITY) {
                SIMULATION.SolidMaskFunction = LID_CAVITY_SOLID_MASK;
                SIMULATION.VelocityBoundaryFunction = LID_CAVITY_FLOW;
                SIMULATION.PressureBoundaryFunction = LID_CAVITY_FLOW_PRESSURE;
            }
            else if(SIMULATION.level == LevelConfiguration::OBSTACLE){
                SIMULATION.SolidMaskFunction = OBSTACLE_SOLID_MASK;
                SIMULATION.VelocityBoundaryFunction = OBSTACLE_FLOW;
                SIMULATION.PressureBoundaryFunction = OBSTACLE_FLOW_PRESSURE;
            }

            else{printf("FAILED LEVEL ASSERTION!\n");}

            SIMULATION.GRID_SOL->SetLevelGeometry(SIMULATION.SolidMaskFunction);
            SIMULATION.GRID_ANT->SetLevelGeometry(SIMULATION.SolidMaskFunction);

            SIMULATION.GRID_SOL->SetGrid(ZERO,ZERO_SCALAR,0);
            SIMULATION.GRID_ANT->SetGrid(ZERO,ZERO_SCALAR,0);
        }

        else if(DIMENSION == 2  && SIM_TYPE == SIM_TYPES::ADI){


            if (SIMULATION.level == LevelConfiguration::STEP) {
                    SIMULATION.SolidMaskFunction2D = BACKWARDS_FACING_STEP_SOLID_MASK_2D;
                    SIMULATION.VelocityBoundaryFunction2D = BACKWARDS_FACING_STEP_2D;
                    SIMULATION.PressureBoundaryFunction2D = BACKWARDS_FACING_STEP_PRESSURE_2D;
                }
            else if(SIMULATION.level == LevelConfiguration::OBSTACLE){
                    SIMULATION.SolidMaskFunction2D = OBSTACLE_SOLID_MASK_2D;
                    SIMULATION.VelocityBoundaryFunction2D = OBSTACLE_FLOW_2D;
                    SIMULATION.PressureBoundaryFunction2D = OBSTACLE_FLOW_PRESSURE_2D;

            }

            else if(SIMULATION.level == LevelConfiguration::LID_CAVITY){
                    SIMULATION.SolidMaskFunction2D = LID_CAVITY_SOLID_MASK_2D;
                    SIMULATION.VelocityBoundaryFunction2D = LID_CAVITY_FLOW_2D;
                    SIMULATION.PressureBoundaryFunction2D = LID_CAVITY_FLOW_PRESSURE_2D;

            }
            else{
                printf("FAILED LEVEL ASSERTION!\n");

            }


            SIMULATION.GRID_SOL->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);
            SIMULATION.GRID_ANT->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);

            SIMULATION.GRID_SOL->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
            SIMULATION.GRID_ANT->SetGrid(ZERO2D,ZERO2D_SCALAR,0);

        }

        else if(DIMENSION == 2 && SIM_TYPE == SIM_TYPES::FLIP){
             if (SIMULATION.level == LevelConfiguration::DAMBREAK) {
                    SIMULATION.SolidMaskFunction2D = DAMBREAK_SOLID_MASK_2D;
                    SIMULATION.VelocityBoundaryFunction2D = DAMBREAK_BORDER_2D;
                    SIMULATION.PressureBoundaryFunction2D = DAMBREAK_PRESSURE_2D;
             }

            else{
                printf("FAILED LEVEL ASSERTION!\n");

            }

            SIMULATION.GRID_SOL->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);
            SIMULATION.GRID_ANT->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);

            SIMULATION.GRID_SOL->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
            SIMULATION.GRID_ANT->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
        }

        else if(DIMENSION == 2 && (SIM_TYPE == SIM_TYPES::FLIP ||SIM_TYPE == SIM_TYPES::SPH)){
             if (SIMULATION.level == LevelConfiguration::DAMBREAK) {
                    SIMULATION.SolidMaskFunction2D = DAMBREAK_SOLID_MASK_2D;
                    SIMULATION.VelocityBoundaryFunction2D = DAMBREAK_BORDER_2D;
                    SIMULATION.PressureBoundaryFunction2D = DAMBREAK_PRESSURE_2D;
             }

            else{
                printf("FAILED LEVEL ASSERTION!\n");

            }

            SIMULATION.GRID_SOL->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);
            SIMULATION.GRID_ANT->SetLevelGeometry(SIMULATION.SolidMaskFunction2D);

            SIMULATION.GRID_SOL->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
            SIMULATION.GRID_ANT->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
        }

        
        else if(DIMENSION == 3 && SIM_TYPE == SIM_TYPES::FLIP){
             if (SIMULATION.level == LevelConfiguration::DAMBREAK) {
                    SIMULATION.SolidMaskFunction = DAMBREAK_SOLID_MASK;
                    SIMULATION.VelocityBoundaryFunction = DAMBREAK_BORDER;
                    SIMULATION.PressureBoundaryFunction = DAMBREAK_PRESSURE;
             }

            else{
                printf("FAILED LEVEL ASSERTION!\n");

            }

            SIMULATION.GRID_SOL->SetLevelGeometry(SIMULATION.SolidMaskFunction);
            SIMULATION.GRID_ANT->SetLevelGeometry(SIMULATION.SolidMaskFunction);

            SIMULATION.GRID_SOL->SetGrid(ZERO,ZERO_SCALAR,0);
            SIMULATION.GRID_ANT->SetGrid(ZERO,ZERO_SCALAR,0);
        }
        

        std::cout << "Simulation config finished!" << std::endl;
    }
};
#endif