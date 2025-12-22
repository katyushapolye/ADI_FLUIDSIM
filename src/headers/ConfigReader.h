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


// Configuration namespace to read simulation parameters
namespace ConfigReader {
    static std::map<std::string, std::string> parameters;

    // Read parameters from a configuration file
    static bool loadFromFile(const std::string& filename) {
        std::ifstream configFile(filename);
        if (!configFile.is_open()) {
            std::cerr << "Error: Could not open configuration file " << filename << std::endl;
            return false;
        }

        parameters.clear();
        std::string line;
        while (std::getline(configFile, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == '/') {
                continue;
            }

            std::istringstream lineStream(line);
            std::string key, value;
            
            // Format expected: key = value
            if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
                // Trim whitespace
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

    // Get parameter value as string
    static std::string getString(const std::string& key, const std::string& defaultValue = "") {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            return it->second;
        }
        return defaultValue;
    }
    
    // Get parameter value as double
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
    
    // Get parameter value as integer
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
        
        if(DIMENSION == 3){
            SIMULATION.GRID_SOL = new MAC();
            SIMULATION.GRID_ANT = new MAC();
            SIMULATION.domain = Domain();
            
            // Configure domain
            SIMULATION.domain.x0 = getDouble("DOMAIN_X0", config.domain.x0);
            SIMULATION.domain.xf = getDouble("DOMAIN_XF", config.domain.xf);
            SIMULATION.domain.y0 = getDouble("DOMAIN_Y0", config.domain.y0);
            SIMULATION.domain.yf = getDouble("DOMAIN_YF", config.domain.yf);
            SIMULATION.domain.z0 = getDouble("DOMAIN_Z0", config.domain.z0);
            SIMULATION.domain.zf = getDouble("DOMAIN_ZF", config.domain.zf);
            
            // Configure simulation parameters
            SIMULATION.dt = getDouble("TIME_STEP", config.dt);
            SIMULATION.RE = getDouble("REYNOLDS_NUMBER", config.RE);
            SIMULATION.EPS = getDouble("VISCOSITY", 0.01);
            SIMULATION.GRID_SIZE = getInt("GRID_SIZE", config.GRID_SIZE);
            SIMULATION.TOLERANCE = getDouble("TOLERANCE", config.TOLERANCE);
            SIMULATION.ExportPath = getString("EXPORT_BASE_PATH", "Exports");
            SIMULATION.CHARACTERISTIC_LENGTH = getDouble("CHARACTERISTIC_LENGTH", config.CHARACTERISTIC_LENGTH);
            SIMULATION.MEAN_VELOCITY = getDouble("MEAN_VELOCITY", config.MEAN_VELOCITY);
            SIMULATION.NEEDS_COMPATIBILITY_CONDITION = getInt("NEEDS_COMPATIBILITY_CONDITION",config.NEEDS_COMPATIBILITY_CONDITION);
            
            
            // Initialize grid
            SIMULATION.GRID_SOL->InitializeGrid(SIMULATION.domain);
            SIMULATION.GRID_ANT->InitializeGrid(SIMULATION.domain);
            
            // Return values from initialization
            SIMULATION.dh = (SIMULATION.GRID_ANT->dh);
            SIMULATION.Nx = SIMULATION.GRID_ANT->Nx;
            SIMULATION.Ny = SIMULATION.GRID_ANT->Ny;
            SIMULATION.Nz = SIMULATION.GRID_ANT->Nz;
            
            // Example of loading level configuration
            std::string levelType = getString("LEVEL_TYPE", "STEP");
            if (levelType == "STEP") {
                SIMULATION.level = LevelConfiguration::STEP;
            } else if (levelType == "CAVITY") {
                SIMULATION.level = LevelConfiguration::LID_CAVITY;
            } 
            else if (levelType == "OBSTACLE"){
                SIMULATION.level = LevelConfiguration::OBSTACLE;
            }
            else if (levelType == "ANALYTICAL"){
                SIMULATION.level = LevelConfiguration::OBSTACLE;
            }
            else {
                std::cerr << "Warning: Unknown level type '" << levelType << "'. Using default." << std::endl;
            }

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

    //2D Simulation
    else{

        SIMULATION2D.GRID_SOL = new MAC2D();
        SIMULATION2D.GRID_ANT = new MAC2D();
        SIMULATION2D.domain =    Domain2D();

        SIMULATION2D.domain.x0 = getDouble("DOMAIN_X0", config.domain.x0);
        SIMULATION2D.domain.xf = getDouble("DOMAIN_XF", config.domain.xf);
        SIMULATION2D.domain.y0 = getDouble("DOMAIN_Y0", config.domain.y0);
        SIMULATION2D.domain.yf = getDouble("DOMAIN_YF", config.domain.yf);
        SIMULATION2D.dt = getDouble("TIME_STEP", config.dt);
        SIMULATION2D.RE = getDouble("REYNOLDS_NUMBER", config.RE);
        SIMULATION2D.EPS = getDouble("VISCOSITY", 0.01);
        SIMULATION2D.GRID_SIZE = getInt("GRID_SIZE", config.GRID_SIZE);
        SIMULATION2D.TOLERANCE = getDouble("TOLERANCE", config.TOLERANCE);
        SIMULATION2D.ExportPath = getString("EXPORT_BASE_PATH", "Exports");
        SIMULATION2D.CHARACTERISTIC_LENGTH = getDouble("CHARACTERISTIC_LENGTH", config.CHARACTERISTIC_LENGTH);
        SIMULATION2D.MEAN_VELOCITY = getDouble("MEAN_VELOCITY", config.MEAN_VELOCITY);

        SIMULATION2D.NEEDS_COMPATIBILITY_CONDITION = getInt("NEEDS_COMPATIBILITY_CONDITION",config.NEEDS_COMPATIBILITY_CONDITION);


        SIMULATION2D.GRID_SOL->InitializeGrid(SIMULATION2D.domain);
        SIMULATION2D.GRID_ANT->InitializeGrid(SIMULATION2D.domain);
        
        SIMULATION2D.dh = (SIMULATION2D.GRID_ANT->dh);
        SIMULATION2D.Nx = SIMULATION2D.GRID_ANT->Nx;
        SIMULATION2D.Ny = SIMULATION2D.GRID_ANT->Ny;

         std::string levelType = getString("LEVEL_TYPE", "STEP");
        if (levelType == "STEP") {
            SIMULATION2D.level = LevelConfiguration::STEP;
        } else if (levelType == "CAVITY") {
            SIMULATION2D.level = LevelConfiguration::LID_CAVITY;
        } 
        else if (levelType == "OBSTACLE"){
            SIMULATION2D.level = LevelConfiguration::OBSTACLE;
        }
        else {
            std::cerr << "Warning: Unknown level type '" << levelType << "'. Using default." << std::endl;
        }

        if (SIMULATION2D.level == LevelConfiguration::STEP) {
                SIMULATION2D.SolidMaskFunction = BACKWARDS_FACING_STEP_SOLID_MASK_2D;
                SIMULATION2D.VelocityBoundaryFunction = BACKWARDS_FACING_STEP_2D;
                SIMULATION2D.PressureBoundaryFunction = BACKWARDS_FACING_STEP_PRESSURE_2D;
            }
        else if(SIMULATION2D.level == LevelConfiguration::OBSTACLE){
                SIMULATION2D.SolidMaskFunction = OBSTACLE_SOLID_MASK_2D;
                SIMULATION2D.VelocityBoundaryFunction = OBSTACLE_FLOW_2D;
                SIMULATION2D.PressureBoundaryFunction = OBSTACLE_FLOW_PRESSURE_2D;

        }

        SIMULATION2D.GRID_SOL->SetLevelGeometry(SIMULATION2D.SolidMaskFunction);
        SIMULATION2D.GRID_ANT->SetLevelGeometry(SIMULATION2D.SolidMaskFunction);

        SIMULATION2D.GRID_SOL->SetGrid(ZERO2D,ZERO2D_SCALAR,0);
        SIMULATION2D.GRID_ANT->SetGrid(ZERO2D,ZERO2D_SCALAR,0);



    }
    }
};
#endif