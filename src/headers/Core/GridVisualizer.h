#ifndef  GRID_VISUALIZER_H
#define GRID_VISUALIZER_H

#include "MAC.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <vector>
#include <string>
#include <cmath>

// Declare external global variable for dimension
extern int DIMENSION;

class GridVisualizer {
private:
    MAC* grid3D;
    MAC* grid2D;
    
    // UI state
    int selectedComponent = 1;  // 0=velocity, 1=u, 2=v, 3=w (3D only), 4=p, 5=solid, 6=divergence, 7=magnitude
    int slicePlane = 0;         // 0=XY, 1=XZ (3D only), 2=YZ (3D only)
    int sliceIndex = 0;
    bool autoScale = true;
    float minValue = 0.0f;
    float maxValue = 1.0f;

    float dh = 0.0f;

    bool twoDimension = true;
    
    // Zoom and pan state
    float zoomLevel = 1.0f;
    
    // Quiver plot data (2D)
    std::vector<float> quiverX;
    std::vector<float> quiverY;
    std::vector<float> quiverU;
    std::vector<float> quiverV;
    
    // Quiver plot data (3D)
    std::vector<float> quiver3DX;
    std::vector<float> quiver3DY;
    std::vector<float> quiver3DZ;
    std::vector<float> quiver3DU;
    std::vector<float> quiver3DV;
    std::vector<float> quiver3DW;
    int stride = 1; // 1 means show every cell, 2 means every other cell, etc.
    
    // Quiver plot settings
    float baseSize = 12.0f;
    ImPlotQuiverFlags quiverFlags = ImPlotQuiverFlags_Colored | ImPlotQuiverFlags_Normalize;
    ImPlot3DQuiverFlags quiver3DFlags = ImPlot3DQuiverFlags_Colored | ImPlot3DQuiverFlags_Normalize;
    ImPlotColormap colormap = ImPlotColormap_Viridis;
    ImPlot3DColormap colormap3D = ImPlot3DColormap_Viridis;
    
    // Heatmap data (for scalar fields)
    std::vector<double> heatmapData;
    int heatmapRows = 0;
    int heatmapCols = 0;
    
    const char* componentNames[8] = {"Velocity Field", "U Velocity", "V Velocity", "W Velocity", "Pressure", "Solid Mask", "Divergence", "Magnitude"};
    const char* planeNames[3] = {"XY Plane", "XZ Plane", "YZ Plane"};

    void ExtractSliceData();
    void ExtractQuiverData();
    void ExtractQuiver3DData();
    void ComputeMinMax();
    
    // Helper methods to get grid dimensions
    int GetNx() const { return (DIMENSION == 2) ? grid2D->Nx : grid3D->Nx; }
    int GetNy() const { return (DIMENSION == 2) ? grid2D->Ny : grid3D->Ny; }
    int GetNz() const { return (DIMENSION == 2) ? 1 : grid3D->Nz; }

    void RenderTelemetryUI();
    void RenderControlUI(int IT, double time, double residual, double frameTime, bool& simulationRunning, bool& stepOnce);
    void RenderExportUI();
    void RenderAerodynamicsUI();

public:
    GridVisualizer(MAC* gridPtr);
    ~GridVisualizer();
    
    void Render(int IT, double time, double residual, double frameTime, bool& simulationRunning, bool& stepOnce);
    void UpdateGrid(MAC* newGrid);



};

// Implementation
GridVisualizer::GridVisualizer(MAC* gridPtr) : grid3D(gridPtr), grid2D(nullptr) {
    if(gridPtr->is2D){

        sliceIndex = 0; // No slicing needed for 2D
        dh = gridPtr->dh;
    }
    else{

    

        sliceIndex = grid3D->Ny / 2; // Start at middle slice
    
        dh = grid3D->dh;
    }
}



GridVisualizer::~GridVisualizer() {}

void GridVisualizer::UpdateGrid(MAC* newGrid) {

    if(newGrid->is2D){
    grid2D = newGrid;
    grid3D = nullptr;
    }
    else{

    
    grid3D = newGrid;
    grid2D = nullptr;
    }
}


void GridVisualizer::ComputeMinMax() {

    // --- Velocity Field (Quiver only) ---
    if (selectedComponent == 0) {
        minValue = 0.0f;
        maxValue = 0.0f;

        if (twoDimension) {
            for (size_t i = 0; i < quiverU.size(); i++) {
                float mag = std::sqrt(quiverU[i]*quiverU[i] + quiverV[i]*quiverV[i]);
                maxValue = (std::max)(maxValue, mag);
            }
        } else {
            for (size_t i = 0; i < quiver3DU.size(); i++) {
                float mag = std::sqrt(
                    quiver3DU[i]*quiver3DU[i] +
                    quiver3DV[i]*quiver3DV[i] +
                    quiver3DW[i]*quiver3DW[i]);
                maxValue = (std::max)(maxValue, mag);
            }
        }

        if (maxValue < 1e-12f) maxValue = 1e-12f;
        return;
    }

    // --- Solid mask ---
    if (selectedComponent == 5) {
        minValue = 0.0f;
        maxValue = 3.0f;
        return;
    }

    // --- Heatmap-based components (Divergence, Magnitude, Pressure, etc.) ---
    if (heatmapData.empty())
        return;

    minValue = (float)heatmapData[0];
    maxValue = (float)heatmapData[0];

    for (double v : heatmapData) {
        minValue = (std::min)(minValue, (float)v);
        maxValue = (std::max)(maxValue, (float)v);
    }

    if (fabs(maxValue - minValue) < 1e-12f)
        maxValue = minValue + 1e-12f;
}


void GridVisualizer::ExtractQuiverData() {
    quiverX.clear();
    quiverY.clear();
    quiverU.clear();
    quiverV.clear();
    
    if (DIMENSION == 2) {
        if (!grid2D) return;
        
        // 2D: Only XY plane, show u and v velocities
        try {
            for (int j = 0; j < grid2D->Ny; j++) {
                for (int i = 0; i < grid2D->Nx; i++) {
                    quiverX.push_back(i + 0.5f);
                    quiverY.push_back(j + 0.5f);
                    
                    // Interpolate u velocity to cell center
                    float u = 0.5f * (grid2D->GetU(j, i) + grid2D->GetU(j, i+1));
                    // Interpolate v velocity to cell center
                    float v = 0.5f * (grid2D->GetV(j, i) + grid2D->GetV(j+1, i));
                    
                    quiverU.push_back(u);
                    quiverV.push_back(v);
                }
            }
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Error in ExtractQuiverData (2D): " << e.what() << std::endl;
            return;
        }
    } else {
        if (!grid3D) return;
        
        int maxSlice = (slicePlane == 0) ? grid3D->Nz - 1 : 
                       (slicePlane == 1) ? grid3D->Ny - 1 : grid3D->Nx - 1;
        sliceIndex = (std::max)(0, (std::min)(sliceIndex, maxSlice));
        
        try {
            switch (slicePlane) {
                case 0: { // XY plane - show u and v velocities
                    int k = (std::min)(sliceIndex, grid3D->Nz - 1);
                    
                    for (int j = 0; j < grid3D->Ny; j++) {
                        for (int i = 0; i < grid3D->Nx; i++) {
                            quiverX.push_back(i + 0.5f);
                            quiverY.push_back(j + 0.5f);
                            
                            float u = 0.5f * (grid3D->GetU(j, i, k) + grid3D->GetU(j, i+1, k));
                            float v = 0.5f * (grid3D->GetV(j, i, k) + grid3D->GetV(j+1, i, k));
                            
                            quiverU.push_back(u);
                            quiverV.push_back(v);
                        }
                    }
                    break;
                }
                
                case 1: { // XZ plane - show u and w velocities
                    int j = (std::min)(sliceIndex, grid3D->Ny - 1);
                    
                    for (int k = 0; k < grid3D->Nz; k++) {
                        for (int i = 0; i < grid3D->Nx; i++) {
                            quiverX.push_back(i + 0.5f);
                            quiverY.push_back(k + 0.5f);
                            
                            float u = 0.5f * (grid3D->GetU(j, i, k) + grid3D->GetU(j, i+1, k));
                            float w = 0.5f * (grid3D->GetW(j, i, k) + grid3D->GetW(j, i, k+1));
                            
                            quiverU.push_back(u);
                            quiverV.push_back(w);
                        }
                    }
                    break;
                }
                
                case 2: { // YZ plane - show v and w velocities
                    int i = (std::min)(sliceIndex, grid3D->Nx - 1);
                    
                    for (int k = 0; k < grid3D->Nz; k++) {
                        for (int j = 0; j < grid3D->Ny; j++) {
                            quiverX.push_back(j + 0.5f);
                            quiverY.push_back(k + 0.5f);
                            
                            float v = 0.5f * (grid3D->GetV(j, i, k) + grid3D->GetV(j+1, i, k));
                            float w = 0.5f * (grid3D->GetW(j, i, k) + grid3D->GetW(j, i, k+1));
                            
                            quiverU.push_back(v);
                            quiverV.push_back(w);
                        }
                    }
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Error in ExtractQuiverData (3D): " << e.what() << std::endl;
            return;
        }
    }
    
    if (autoScale) {
        ComputeMinMax();
    }
}

void GridVisualizer::ExtractQuiver3DData() {
    quiver3DX.clear();
    quiver3DY.clear();
    quiver3DZ.clear();
    quiver3DU.clear();
    quiver3DV.clear();
    quiver3DW.clear();
    
    if (DIMENSION == 2 || !grid3D) return;
    
    try {
        for (int k = 0; k < grid3D->Nz; k += stride) {
            for (int j = 0; j < grid3D->Ny; j += stride) {
                for (int i = 0; i < grid3D->Nx; i += stride) {
                    quiver3DX.push_back(i + 0.5f);
                    quiver3DY.push_back(j + 0.5f);
                    quiver3DZ.push_back(k + 0.5f);
                    
                    // Interpolate velocities to cell center
                    float u = 0.5f * (grid3D->GetU(j, i, k) + grid3D->GetU(j, i+1, k));
                    float v = 0.5f * (grid3D->GetV(j, i, k) + grid3D->GetV(j+1, i, k));
                    float w = 0.5f * (grid3D->GetW(j, i, k) + grid3D->GetW(j, i, k+1));
                    
                    quiver3DU.push_back(u);
                    quiver3DV.push_back(v);
                    quiver3DW.push_back(w);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "[EXCEPTION] Error in ExtractQuiver3DData: " << e.what() << std::endl;
        return;
    }
    
    if (autoScale) {
        ComputeMinMax();
    }
}

void GridVisualizer::ExtractSliceData() {
    heatmapData.clear();
    
    if (DIMENSION == 2) {
        if (!grid2D) {
            std::cout << "[ERROR] Grid2D is null!" << std::endl;
            return;
        }
        
        try {
            if (selectedComponent == 1) { // U
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx + 1;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y < grid2D->Ny; y++) {
                    for (int x = 0; x <= grid2D->Nx; x++) {
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetU(y, x);
                    }
                }
            } else if (selectedComponent == 2) { // V
                heatmapRows = grid2D->Ny + 1;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y <= grid2D->Ny; y++) {
                    for (int x = 0; x < grid2D->Nx; x++) {
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetV(y, x);
                    }
                }
            } else if (selectedComponent == 4) { // Pressure
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y < grid2D->Ny; y++) {
                    for (int x = 0; x < grid2D->Nx; x++) {
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetP(y, x);
                    }
                }
            } else if (selectedComponent == 5) { // Solid Mask
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y < grid2D->Ny; y++) {
                    for (int x = 0; x < grid2D->Nx; x++) {
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetSolid(y, x);
                    }
                }
            } else if (selectedComponent == 6) { // Divergence
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y < grid2D->Ny; y++) {
                    for (int x = 0; x < grid2D->Nx; x++) {
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetDivergencyAt(y, x);
                    }
                }
            } else if (selectedComponent == 7) { // Magnitude
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);
                
                for (int y = 0; y < grid2D->Ny; y++) {
                    for (int x = 0; x < grid2D->Nx; x++) {
                        // Interpolate velocities to cell center
                        float u = 0.5f * (grid2D->GetU(y, x) + grid2D->GetU(y, x+1));
                        float v = 0.5f * (grid2D->GetV(y, x) + grid2D->GetV(y+1, x));
                        float mag = std::sqrt(u*u + v*v);
                        
                        int idx = y * heatmapCols + x;
                        heatmapData[idx] = mag;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Error in ExtractSliceData (2D): " << e.what() << std::endl;
            return;
        }
    } else {
        if (!grid3D) {
            std::cout << "[ERROR] Grid3D is null!" << std::endl;
            return;
        }
        
        int maxSlice = (slicePlane == 0) ? grid3D->Nz - 1 : 
                       (slicePlane == 1) ? grid3D->Ny - 1 : grid3D->Nx - 1;
        sliceIndex = (std::max)(0, (std::min)(sliceIndex, maxSlice));
        
        try {
            switch (slicePlane) {
                case 0: { // XY plane
                    int k = (std::min)(sliceIndex, grid3D->Nz - 1);
                    
                    if (selectedComponent == 1) { // U
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx + 1;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x <= grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetU(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 2) { // V
                        heatmapRows = grid3D->Ny + 1;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y <= grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetV(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 3) { // W
                        k = (std::min)(sliceIndex, grid3D->Nz);
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetW(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 4) { // Pressure
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetP(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 5) { // Solid Mask
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetSolid(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 6) { // Divergence
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetDivergencyAt(y, x, k);
                            }
                        }
                    } else if (selectedComponent == 7) { // Magnitude
                        heatmapRows = grid3D->Ny;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int y = 0; y < grid3D->Ny; y++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                float u = 0.5f * (grid3D->GetU(y, x, k) + grid3D->GetU(y, x+1, k));
                                float v = 0.5f * (grid3D->GetV(y, x, k) + grid3D->GetV(y+1, x, k));
                                float w = 0.5f * (grid3D->GetW(y, x, k) + grid3D->GetW(y, x, k+1));
                                float mag = std::sqrt(u*u + v*v + w*w);
                                
                                int idx = y * heatmapCols + x;
                                heatmapData[idx] = mag;
                            }
                        }
                    }
                    break;
                }
                
                case 1: { // XZ plane
                    int y = (std::min)(sliceIndex, grid3D->Ny - 1);
                    
                    if (selectedComponent == 1) { // U
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx + 1;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x <= grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetU(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 2) { // V
                        y = (std::min)(sliceIndex, grid3D->Ny);
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetV(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 3) { // W
                        heatmapRows = grid3D->Nz + 1;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z <= grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetW(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 4) { // Pressure
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetP(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 5) { // Solid Mask
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetSolid(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 6) { // Divergence
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = grid3D->GetDivergencyAt(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 7) { // Magnitude
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Nx;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int x = 0; x < grid3D->Nx; x++) {
                                float u = 0.5f * (grid3D->GetU(y, x, z) + grid3D->GetU(y, x+1, z));
                                float v = 0.5f * (grid3D->GetV(y, x, z) + grid3D->GetV(y+1, x, z));
                                float w = 0.5f * (grid3D->GetW(y, x, z) + grid3D->GetW(y, x, z+1));
                                float mag = std::sqrt(u*u + v*v + w*w);
                                
                                int idx = z * heatmapCols + x;
                                heatmapData[idx] = mag;
                            }
                        }
                    }
                    break;
                }
                
                case 2: { // YZ plane
                    int x = (std::min)(sliceIndex, grid3D->Nx - 1);
                    
                    if (selectedComponent == 1) { // U
                        x = (std::min)(sliceIndex, grid3D->Nx);
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetU(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 2) { // V
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny + 1;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y <= grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetV(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 3) { // W
                        heatmapRows = grid3D->Nz + 1;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z <= grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetW(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 4) { // Pressure
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetP(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 5) { // Solid Mask
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetSolid(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 6) { // Divergence
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = grid3D->GetDivergencyAt(y, x, z);
                            }
                        }
                    } else if (selectedComponent == 7) { // Magnitude
                        heatmapRows = grid3D->Nz;
                        heatmapCols = grid3D->Ny;
                        heatmapData.resize(heatmapRows * heatmapCols);
                        
                        for (int z = 0; z < grid3D->Nz; z++) {
                            for (int y = 0; y < grid3D->Ny; y++) {
                                float u = 0.5f * (grid3D->GetU(y, x, z) + grid3D->GetU(y, x+1, z));
                                float v = 0.5f * (grid3D->GetV(y, x, z) + grid3D->GetV(y+1, x, z));
                                float w = 0.5f * (grid3D->GetW(y, x, z) + grid3D->GetW(y, x, z+1));
                                float mag = std::sqrt(u*u + v*v + w*w);
                                
                                int idx = z * heatmapCols + y;
                                heatmapData[idx] = mag;
                            }
                        }
                    }
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Error in ExtractSliceData (3D): " << e.what() << std::endl;
            return;
        }
    }
    
    if (autoScale) {
        ComputeMinMax();
    }
}

void GridVisualizer::Render(int IT,  double time, double residual, double frameTime, bool& simulationRunning, bool& stepOnce) {
    if ((DIMENSION == 2 && !grid2D) || (DIMENSION == 3 && !grid3D)) return;
    
    ImGui::Begin("Grid Visualizer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    if (ImGui::BeginTabBar("VisualizationTabs")) {
        if (ImGui::BeginTabItem("2D Slice")) {
            twoDimension = true;
            
            // Component selection dropdown
            ImGui::Text("Component:");
            ImGui::SetNextItemWidth(200);
            const char* componentLabels[8] = {
                "Velocity Field",
                "U Velocity",
                "V Velocity",
                "W Velocity",
                "Pressure",
                "Solid Mask",
                "Divergence",
                "Magnitude"
            };
            
            // Filter out W component for 2D
            if (DIMENSION == 2) {
                const char* componentLabels2D[7] = {
                    "Velocity Field", "U Velocity", "V Velocity",
                    "Pressure", "Solid Mask", "Divergence", "Magnitude"
                };
                int displayIndex = selectedComponent;
                if (selectedComponent > 3) displayIndex--;
                
                if (ImGui::Combo("##Component", &displayIndex, componentLabels2D, 7)) {
                    selectedComponent = (displayIndex >= 3) ? displayIndex + 1 : displayIndex;
                }
            } else {
                ImGui::Combo("##Component", &selectedComponent, componentLabels, 8);
            }
            
            ImGui::Separator();
            
            // Plane selection - only for 3D
            if (DIMENSION == 3) {
                ImGui::Text("Slice Plane:");
                ImGui::RadioButton("XY (k)", &slicePlane, 0); ImGui::SameLine();
                ImGui::RadioButton("XZ (j)", &slicePlane, 1); ImGui::SameLine();
                ImGui::RadioButton("YZ (i)", &slicePlane, 2);
                
                // Slice index slider
                int maxSlice = (slicePlane == 0) ? grid3D->Nz - 1 : 
                               (slicePlane == 1) ? grid3D->Ny - 1 : grid3D->Nx - 1;
                ImGui::SliderInt("Slice Index", &sliceIndex, 0, maxSlice);
                
                // Display physical position of slice
                float slicePosition = sliceIndex * dh;
                ImGui::Text("Slice Position: %.3f", slicePosition);
                
                ImGui::Separator();
            } else {
                // Force XY plane for 2D
                slicePlane = 0;
                sliceIndex = 0;
            }
            
            // Controls based on component type
            if (selectedComponent == 0) {
                // Quiver plot controls
                ImGui::Text("Quiver Plot Settings:");
                
                if (ImPlot::ColormapButton(ImPlot::GetColormapName(colormap), ImVec2(150, 0), colormap)) {
                    colormap = (ImPlotColormap)((colormap + 1) % ImPlot::GetColormapCount());
                }
                ImGui::SameLine();
                ImGui::Checkbox("Auto Scale", &autoScale);
                if (!autoScale) {
                    ImGui::SetNextItemWidth(225);
                    ImGui::DragFloatRange2("Min / Max", &minValue, &maxValue, 0.01f, -20, 20, 
                                          nullptr, nullptr, ImGuiSliderFlags_AlwaysClamp);
                    if (maxValue <= minValue + 0.01f) {
                        maxValue = minValue + 0.01f;
                    }
                }
                
                ImGui::SetNextItemWidth(225);
                ImGui::DragFloat("Arrow Size", &baseSize, 0.1f, 0, 100);
                
                ImGui::CheckboxFlags("Normalize", (unsigned int*)&quiverFlags, ImPlotQuiverFlags_Normalize);
                ImGui::CheckboxFlags("Color Coded", (unsigned int*)&quiverFlags, ImPlotQuiverFlags_Colored);
            } else {
                // Heatmap controls
                if (selectedComponent == 5) {
                    // For solid mask, disable auto scale and fix range
                    ImGui::Text("Cell Types: 0=Fluid, 1=Solid, 2=Empty, 3=Inflow");
                    autoScale = false;
                    minValue = 0.0f;
                    maxValue = 3.0f;
                }
                else {
                    ImGui::Checkbox("Auto Scale", &autoScale);
                    if (!autoScale) {
                        ImGui::SliderFloat("Min Value", &minValue, -10.0f, 10.0f);
                        ImGui::SliderFloat("Max Value", &maxValue, -10.0f, 10.0f);
                    }
                }
            }
            
            ImGui::Separator();
            
            // Zoom control
            ImGui::Text("View Controls:");
            ImGui::SliderFloat("Zoom", &zoomLevel, 0.1f, 5.0f, "%.1fx");
            ImGui::SameLine();
            if (ImGui::Button("Reset Zoom")) {
                zoomLevel = 1.0f;
            }
            ImGui::TextDisabled("(Use mouse wheel to zoom, drag to pan)");
            
            // Extract data
            if (selectedComponent == 0) {
                ExtractQuiverData();
            } else {
                ExtractSliceData();
            }
            
            // Display info
            ImGui::Separator();
            if (DIMENSION == 2) {
                float physicalWidth = GetNx() * dh;
                float physicalHeight = GetNy() * dh;
                ImGui::Text("Grid: %dx%d (2D)", GetNx(), GetNy());
                ImGui::Text("Physical Size: %.3f x %.3f", physicalWidth, physicalHeight);
            } else {
                float physicalX = GetNx() * dh;
                float physicalY = GetNy() * dh;
                float physicalZ = GetNz() * dh;
                ImGui::Text("Grid: %dx%dx%d (3D)", GetNx(), GetNy(), GetNz());
                ImGui::Text("Physical Size: %.3f x %.3f x %.3f", physicalX, physicalY, physicalZ);
            }
            
            if (selectedComponent == 0) {
                ImGui::Text("Vectors: %zu", quiverX.size());
            } else {
                ImGui::Text("Heatmap: %dx%d", heatmapCols, heatmapRows);
            }
            ImGui::Text("Range: [%.6f, %.6f]", minValue, maxValue);
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("3D Volume")) {
            twoDimension = false;
            if (DIMENSION == 2) {
                ImGui::Text("3D visualization not available in 2D mode.");
            } else {
                ImGui::Text("3D Quiver Plot Settings:");
                
                ImGui::SetNextItemWidth(225);
                if (ImGui::SliderInt("Stride", &stride, 1, 10)) {

                }
                ImGui::SameLine();
                ImGui::TextDisabled("(Higher = Fewer arrows)");
                ImGui::SameLine();
                ImGui::Text("Colormap");
                if (ImPlot::ColormapButton(ImPlot3D::GetColormapName(colormap3D),ImVec2(225,0),colormap3D)) {
                    colormap3D = (colormap3D + 1) % ImPlot3D::GetColormapCount();

                }
                ImGui::Checkbox("Auto Scale", &autoScale);
                if (!autoScale) {
                    ImGui::SetNextItemWidth(225);
                    ImGui::DragFloatRange2("Min / Max", &minValue, &maxValue, 0.01f, -20, 20, 
                                          nullptr, nullptr, ImGuiSliderFlags_AlwaysClamp);
                    if (maxValue <= minValue + 0.01f) {
                        maxValue = minValue + 0.01f;
                    }
                }
                
                ImGui::SetNextItemWidth(225);
                ImGui::DragFloat("Arrow Size", &baseSize, 0.1f, 0, 100);
                
                ImGui::CheckboxFlags("Normalize", (unsigned int*)&quiver3DFlags, ImPlot3DQuiverFlags_Normalize);
                ImGui::CheckboxFlags("Color Coded", (unsigned int*)&quiver3DFlags, ImPlot3DQuiverFlags_Colored);
                
                ImGui::Separator();
                
                // Extract 3D quiver data
                ExtractQuiver3DData();
                
                float physicalX = GetNx() * dh;
                float physicalY = GetNy() * dh;
                float physicalZ = GetNz() * dh;
                ImGui::Text("Grid: %dx%dx%d", GetNx(), GetNy(), GetNz());
                ImGui::Text("Physical Size: %.3f x %.3f x %.3f", physicalX, physicalY, physicalZ);
                ImGui::Text("Vectors: %zu", quiver3DX.size());
                ImGui::Text("Range: [%.6f, %.6f]", minValue, maxValue);
            }
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
    
    ImGui::End();

    // Plot window with scrollable child region
    if(twoDimension){
        ImGui::Begin("2D Visualization", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        // Use a discrete colormap for solid mask with exactly 4 colors
        if (selectedComponent == 5) {
            // Create custom discrete colormap for cell types
            static bool colormapCreated = false;
            static ImPlotColormap solidColormap;
            if (!colormapCreated) {
                ImVec4 colors[4] = {
                    ImVec4(0.2f, 0.4f, 0.8f, 1.0f),  // 0 = FLUID_CELL  (blue)
                    ImVec4(0.3f, 0.3f, 0.3f, 1.0f),  // 1 = SOLID_CELL  (gray)
                    ImVec4(0.9f, 0.9f, 0.9f, 1.0f),  // 2 = EMPTY_CELL  (light gray)
                    ImVec4(0.8f, 0.2f, 0.2f, 1.0f)   // 3 = INFLOW_CELL (red)
                };
                solidColormap = ImPlot::AddColormap("SolidMask", colors, 4);
                colormapCreated = true;
            }
            ImPlot::PushColormap(solidColormap);
        } else {
            ImPlot::PushColormap(colormap);
        }

        // Calculate physical dimensions
        float width = 1.0f, height = 1.0f;
        
        if (DIMENSION == 2) {
            // 2D mode: always XY plane
            width  = (float)GetNx() * dh;
            height = (float)GetNy() * dh;
        } else {
            // 3D mode: depends on slice plane
            switch(slicePlane) {
                case 0: // XY plane
                    width  = (float)GetNx() * dh; // X axis
                    height = (float)GetNy() * dh; // Y axis
                    break;
                case 1: // XZ plane
                    width  = (float)GetNx() * dh; // X axis
                    height = (float)GetNz() * dh; // Z axis
                    break;
                case 2: // YZ plane
                    width  = (float)GetNy() * dh; // Y axis
                    height = (float)GetNz() * dh; // Z axis
                    break;
            }
        }

        // Calculate plot size based on zoom and aspect ratio
        float aspectRatio = width / height;
        float baseHeight = 600.0f;
        float plotHeight = baseHeight * zoomLevel;
        float plotWidth  = plotHeight * aspectRatio;
        
        // Get available content region
        ImVec2 availRegion = ImGui::GetContentRegionAvail();
        
        // Create scrollable child window if plot is larger than available space
        bool needsScrolling = (plotWidth > availRegion.x - 20) || (plotHeight > availRegion.y - 20);
        
        if (needsScrolling) {
            // Create child window with scrollbars
            ImGui::BeginChild("ScrollRegion", ImVec2(0, 0), false, 
                            ImGuiWindowFlags_HorizontalScrollbar);
        }

        // --- Quiver Plot ---
        if (selectedComponent == 0 && !quiverX.empty()) {
            if (ImPlot::BeginPlot("##QuiverPlot", ImVec2(plotWidth, plotHeight))) {
                // Set axis limits according to physical dimensions
                ImPlot::SetupAxes(
                    (DIMENSION == 2 || slicePlane == 0 || slicePlane == 1) ? "X" : "Y",
                    (DIMENSION == 2 || slicePlane == 0) ? "Y" : "Z"
                );

                ImPlot::SetupAxisLimits(ImAxis_X1, 0, grid2D->Nx, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, grid2D->Ny, ImGuiCond_Always);
            
                ImPlot::SetNextQuiverStyle(baseSize, ImPlot::GetColormapColor(1));
                ImPlot::PlotQuiver("Velocity", 
                                   quiverX.data(), quiverY.data(), 
                                   quiverU.data(), quiverV.data(), 
                                   quiverX.size(), 
                                   minValue, maxValue, 
                                   quiverFlags);
                ImPlot::EndPlot();
            }
        }

        // --- Heatmap Plot ---
        else if (selectedComponent > 0 && !heatmapData.empty()) {
            if (ImPlot::BeginPlot("##Heatmap", ImVec2(plotWidth, plotHeight))) {
                ImPlot::SetupAxes(nullptr, nullptr, 0);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, width, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, height, ImGuiCond_Always);
            
                ImPlot::PlotHeatmap(componentNames[selectedComponent], 
                                    heatmapData.data(), 
                                    heatmapRows, 
                                    heatmapCols,
                                    minValue,
                                    maxValue,
                                    nullptr,
                                    ImPlotPoint(0, 0),
                                    ImPlotPoint(width, height));
                
                ImPlot::EndPlot();
            }
        
            ImGui::SameLine();

            ImPlot::ColormapScale("##HeatmapScale", minValue, maxValue, ImVec2(60, plotHeight));
        }

        if (needsScrolling) {
            ImGui::EndChild();
        }

        ImPlot::PopColormap();

        ImGui::End();
    } else {
// 3D Volume Visualization
if (DIMENSION == 3 && !quiver3DX.empty()) {
    ImGui::Begin("3D Visualization", nullptr, ImGuiWindowFlags_NoScrollbar);
    ImPlot::PushColormap(colormap3D);
    ImPlot3D::PushColormap(colormap3D);
    float plotSize = ImGui::GetTextLineHeight() * 63;
    
    // Calculate physical domain size
    float Lx = grid3D->Nx ;
    float Ly = grid3D->Ny ;
    float Lz = grid3D->Nz ;
    
    if (ImPlot3D::BeginPlot("Quiver Plot 3D", ImVec2(plotSize, plotSize))) {
        // Set up axis limits based on physical dimensions
        ImPlot3D::SetupAxisLimits(ImAxis3D_X, 0, Lx, ImGuiCond_Always);
        ImPlot3D::SetupAxisLimits(ImAxis3D_Y, 0, Ly, ImGuiCond_Always);
        ImPlot3D::SetupAxisLimits(ImAxis3D_Z, 0, Lz, ImGuiCond_Always);
        

        
        ImPlot3D::SetupAxes("X", "Z", "Y");
        ImPlot3D::SetNextQuiverStyle(baseSize, ImPlot3D::GetColormapColor(1)); 
        
        ImPlot3D::PlotQuiver("Magnitude", 
            quiver3DX.data(), quiver3DZ.data(), quiver3DY.data(),
            quiver3DU.data(), quiver3DW.data(), quiver3DV.data(),
            quiver3DX.size(), 
            minValue, maxValue, 
            quiver3DFlags);
        
        ImPlot3D::EndPlot();
    }
    
    ImGui::SameLine();
    ImPlot::ColormapScale("##HeatmapScale", minValue, maxValue, ImVec2(100, plotSize));
    ImPlot::PopColormap();
    ImPlot3D::PopColormap();
    ImGui::End();
    }
}


    this->RenderControlUI(IT,time,residual,frameTime,simulationRunning,stepOnce);
    this->RenderTelemetryUI();
    this->RenderAerodynamicsUI();
    this->RenderExportUI();
}

void GridVisualizer::RenderControlUI(int IT,  double time, double residual, double frameTime, bool& simulationRunning, bool& stepOnce) {
    ImGui::Begin("Simulation Control");
    
    ImGui::Text("Iteration: %d" , IT);
    ImGui::Text("Time: %.4f", time);
    ImGui::Text("Residual: %.8f", residual);
    ImGui::Text("Timestep: %.8f", SIMULATION.dt);
    ImGui::Text("Divergence: %.6e", SIMULATION.GRID_SOL->GetDivSum());

    
    
    ImGui::Text("Frame Time: %.8f ms", frameTime * 1000.0);
    
    if (simulationRunning) {
        ImGui::Text("Status: Running");
    } else {
        ImGui::Text("Status: Paused");
    }
    
    ImGui::Separator();
    
    if (ImGui::Button(simulationRunning ? "Pause" : "Run")) {
        simulationRunning = !simulationRunning;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        stepOnce = true;
    }
    
    ImGui::End();
}

void GridVisualizer::RenderTelemetryUI()
{

    ImGui::Begin("Simulation Telemetry");
    
    const double t = TELEMETRY.time.back();
    const double window = 5.0;
    float avail = ImGui::GetContentRegionAvail().x;
    float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    ImVec2 plot_size(plot_w, 300);
    
    // ---- Row 1 ----
    if (ImPlot::BeginPlot("Divergence Evolution", plot_size)) {
        ImPlot::SetupAxes("Time", "Divergence Sum",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        ImPlot::PlotLine("After Projection",
            TELEMETRY.time.data(), TELEMETRY.div_sum.data(), TELEMETRY.time.size());
        ImPlot::PlotLine("Before Projection",
            TELEMETRY.time.data(), TELEMETRY.div_sum_before_proj.data(), TELEMETRY.time.size());
        ImPlot::EndPlot();
    }
    
    ImGui::SameLine();
    
    if (ImPlot::BeginPlot("CFL Condition", plot_size)) {
        ImPlot::SetupAxes("Time", "CFL",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );



        ImPlot::PlotLine("Adv. CFL",
            TELEMETRY.time.data(), TELEMETRY.advcfl.data(), TELEMETRY.time.size());
        ImPlot::PlotLine("Diff. CFL",
            TELEMETRY.time.data(), TELEMETRY.diffcfl.data(), TELEMETRY.time.size());
        double x = 1.0;
        ImPlot::PlotInfLines("CFL = 1", &x, 0,1);
        ImPlot::EndPlot();
    }
    
    // ---- Row 2 ----
    if (ImPlot::BeginPlot("Grid Residual", plot_size)) {
        ImPlot::SetupAxes(
            "Time", "Residual",
            ImPlotAxisFlags_None,
            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit
        );



        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );

        ImPlot::PlotLine(
            "Iteration Residual",
            TELEMETRY.time.data(),
            TELEMETRY.residual.data(),
            TELEMETRY.time.size()
        );

        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    
    if (ImPlot::BeginPlot("Performance (CPU vs GPU)", plot_size)) {
        double xs[] = {0.0, 1.0};
        static const char* labels[] = {"ADI (CPU)", "Pressure (GPU)"};
        double values[] = {
            TELEMETRY.cpu_time.back(),
            TELEMETRY.gpu_time.back()
        };
    
        static double y_max = 0.05;
        y_max = (std::max)(y_max, 1.2 * (std::max)(values[0], values[1]));
    
        ImPlot::SetupAxes(nullptr, "Seconds");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, y_max, ImGuiCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, xs, 2, labels);
        ImVec4 colors[] = { ImVec4(0.4f, 0.7f, 0.0f, 1.0f),
                            ImVec4(0.2f, 0.4f, 1.0f, 1.0f) };

        for (int i = 0; i < 2; i++) {
            ImPlot::PushStyleColor(ImPlotCol_Fill, colors[i]);
            ImPlot::PlotBars(labels[i], &xs[i], &values[i], 1, 0.5);
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }
    
    ImGui::End();
}

void GridVisualizer::RenderAerodynamicsUI() {
    ImGui::Begin("Aerodynamics Telemetry");

    const double t = TELEMETRY.time.back();
    const double window = 5.0;
    
    float avail = ImGui::GetContentRegionAvail().x;
    float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    ImVec2 plot_size(plot_w, 300);
    
    // ---- Row 1: Lift and Drag Coefficients ----
    if (ImPlot::BeginPlot("Lift Coefficient (Cl)", plot_size)) {
        ImPlot::SetupAxes("Time", "Cl",
                         ImPlotAxisFlags_None,
                         ImPlotAxisFlags_AutoFit);
        
        if (!AERODYNAMICS.Cl.empty()) {
            ImPlot::PushStyleColor(ImPlotCol_Line,ImVec4(0.0,0.0,1.0,1.0f));

        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );
            ImPlot::PlotLine("Cl",
                           AERODYNAMICS.time.data(), 
                           AERODYNAMICS.Cl.data(), 
                           AERODYNAMICS.Cl.size());
            ImPlot::PopStyleColor();
            
            // Show current value
            double current_cl = AERODYNAMICS.Cl.back();
            ImPlot::PlotText(("Cl = " + std::to_string(current_cl)).c_str(), 
                           AERODYNAMICS.time.back(), current_cl);
        }
        ImPlot::EndPlot();
    }
    
    ImGui::SameLine();
    
    if (ImPlot::BeginPlot("Drag Coefficient (Cd)", plot_size)) {
        ImPlot::SetupAxes("Time", "Cd",
                         ImPlotAxisFlags_None,
                         ImPlotAxisFlags_AutoFit);
        
        if (!AERODYNAMICS.Cd.empty()) {
            ImPlot::PushStyleColor(ImPlotCol_Line,ImVec4(1.0,0.0,0.0,1.0f));
                    ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
            );

            ImPlot::PlotLine("Cd",
                           AERODYNAMICS.time.data(), 
                           AERODYNAMICS.Cd.data(), 
                           AERODYNAMICS.Cd.size());
            ImPlot::PopStyleColor();
            // Show current value
            double current_cd = AERODYNAMICS.Cd.back();
            ImPlot::PlotText(("Cd = " + std::to_string(current_cd)).c_str(), 
                           AERODYNAMICS.time.back(), current_cd);
        }
        ImPlot::EndPlot();
    }
    
    // ---- Row 2: Pressure Difference and Lift/Drag Ratio ----
    if (ImPlot::BeginPlot("Pressure Difference", plot_size)) {

        ImPlot::SetupAxes("Time", "DeltaP",
                         ImPlotAxisFlags_None,
                         ImPlotAxisFlags_AutoFit);
        
        if (!AERODYNAMICS.pressure_drop.empty()) {
            ImPlot::PushStyleColor(ImPlotCol_Line,ImVec4(0.0,1.0,0.0,1.0f));
                                ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );
            ImPlot::PlotLine("DeltaP",
                           AERODYNAMICS.time.data(), 
                           AERODYNAMICS.pressure_drop.data(), 
                           AERODYNAMICS.pressure_drop.size());
            ImPlot::PopStyleColor();
            
            // Show current value
            double current_pdiff = AERODYNAMICS.pressure_drop.back();
            ImPlot::PlotText(("DeltaP = " + std::to_string(current_pdiff)).c_str(), 
                           AERODYNAMICS.time.back(), current_pdiff);
        }
        ImPlot::EndPlot();
    }
    
    ImGui::SameLine();
    
    if (ImPlot::BeginPlot("Lift-to-Drag Ratio", plot_size)) {
        ImPlot::SetupAxes("Time", "Cl/Cd",
                         ImPlotAxisFlags_None,
                         ImPlotAxisFlags_AutoFit);
        
        if (!AERODYNAMICS.Cl.empty() && !AERODYNAMICS.Cd.empty()) {

        
            ImPlot::PushStyleColor(ImPlotCol_Line,ImVec4(1.0,1.0,1.0,1.0f));
                                ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.5,
            t+window*0.5,
            ImGuiCond_Always
        );
            ImPlot::PlotLine("L/D",
                           AERODYNAMICS.time.data(), 
                           AERODYNAMICS.LtoDRatio.data(), 
                           AERODYNAMICS.LtoDRatio.size());
            ImPlot::PopStyleColor();
            

        }
        ImPlot::EndPlot();
    }
    
    // ---- Row 3: Statistics Table ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Current Aerodynamic Data:");
    ImGui::Spacing();
    if (ImGui::BeginTable("AeroStatsTable", 4, 
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Parameter");
        ImGui::TableSetupColumn("Current");
        ImGui::TableSetupColumn("Mean (last 100)");
        ImGui::TableSetupColumn("Std Dev (last 100)");
        ImGui::TableHeadersRow();
        
        auto compute_stats = [](const std::vector<double>& data, int n_samples) {
            if (data.empty()) return std::make_pair(0.0, 0.0);
            
            int start = (std::max)(0, (int)data.size() - n_samples);
            int count = data.size() - start;
            
            double sum = 0.0;
            for (int i = start; i < data.size(); i++) {
                sum += data[i];
            }
            double mean = sum / count;
            
            double sq_sum = 0.0;
            for (int i = start; i < data.size(); i++) {
                double diff = data[i] - mean;
                sq_sum += diff * diff;
            }
            double std_dev = std::sqrt(sq_sum / count);
            
            return std::make_pair(mean, std_dev);
        };
        
        // Lift coefficient row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Cl");
        ImGui::TableNextColumn();
        if (!AERODYNAMICS.Cl.empty()) {
            ImGui::Text("%.6f", AERODYNAMICS.Cl.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(AERODYNAMICS.Cl, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        } else {
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
        }
        
        // Drag coefficient row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Cd");
        ImGui::TableNextColumn();
        if (!AERODYNAMICS.Cd.empty()) {
            ImGui::Text("%.6f", AERODYNAMICS.Cd.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(AERODYNAMICS.Cd, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        } else {
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
        }
        
        // Pressure difference row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("ΔP");
        ImGui::TableNextColumn();
        if (!AERODYNAMICS.pressure_drop.empty()) {
            ImGui::Text("%.6f", AERODYNAMICS.pressure_drop.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(AERODYNAMICS.pressure_drop, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        } else {
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
        }
        
        // L/D ratio row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("L/D");
        ImGui::TableNextColumn();
        if (!AERODYNAMICS.Cl.empty() && !AERODYNAMICS.Cd.empty()) {
            double cd = AERODYNAMICS.Cd.back();
            double ld = (std::abs(cd) > 1e-10) ? AERODYNAMICS.Cl.back() / cd : 0.0;
            ImGui::Text("%.6f", ld);
            
            // Compute stats for L/D
            std::vector<double> ld_ratio;
            int n_samples = (std::min)(100, (int)AERODYNAMICS.Cl.size());
            int start = AERODYNAMICS.Cl.size() - n_samples;
            
            for (int i = start; i < AERODYNAMICS.Cl.size(); i++) {
                double cd_val = AERODYNAMICS.Cd[i];
                if (std::abs(cd_val) > 1e-10) {
                    ld_ratio.push_back(AERODYNAMICS.Cl[i] / cd_val);
                }
            }
            
            if (!ld_ratio.empty()) {
                auto stats = compute_stats(ld_ratio, ld_ratio.size());
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", stats.first);
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", stats.second);
            } else {
                ImGui::TableNextColumn(); ImGui::Text("-");
                ImGui::TableNextColumn(); ImGui::Text("-");
            }
        } else {
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
            ImGui::TableNextColumn(); ImGui::Text("-");
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}

void GridVisualizer::RenderExportUI() {
    ImGui::Begin("Export Settings");
    
    bool wasEnabled = EXPORT_SETTINGS.enabled;
    ImGui::Checkbox("Enable Export", &EXPORT_SETTINGS.enabled);
    

    
    if (EXPORT_SETTINGS.enabled) {
        ImGui::Separator();
        ImGui::Text("Export Intervals");
        ImGui::SliderInt("Telemetry (Every N Frames)", &EXPORT_SETTINGS.telemetryInterval, 1, 100);
        ImGui::SliderInt("Grid (Every N Frames)", &EXPORT_SETTINGS.gridInterval, 1, 500);
        
        ImGui::Separator();
        ImGui::Text("Grid Export");
        ImGui::Checkbox("Export Grid (VTK)", &EXPORT_SETTINGS.exportGridData);
        
        ImGui::Separator();
        ImGui::Text("Simulation Telemetry");
        ImGui::Checkbox("Time", &EXPORT_SETTINGS.exportTime);
        ImGui::Checkbox("Divergence Sum", &EXPORT_SETTINGS.exportDivSum);
        ImGui::Checkbox("Divergence Before Projection", &EXPORT_SETTINGS.exportDivSumBeforeProj);
        ImGui::Checkbox("CFL Number", &EXPORT_SETTINGS.exportCFL);
        ImGui::Checkbox("Residual", &EXPORT_SETTINGS.exportResidual);
        ImGui::Checkbox("CPU Time (ADI)", &EXPORT_SETTINGS.exportCPUTime);
        ImGui::Checkbox("GPU Time (Pressure)", &EXPORT_SETTINGS.exportGPUTime);
        
        // Show aerodynamics options only for 2D with obstacle
        if ( SIMULATION.level == LevelConfiguration::OBSTACLE || SIMULATION.level == LevelConfiguration::OBSTACLE )  {
            ImGui::Separator();
            ImGui::Text("Aerodynamics Telemetry");
            ImGui::Checkbox("Lift Coefficient (Cl)", &EXPORT_SETTINGS.exportCl);
            ImGui::Checkbox("Drag Coefficient (Cd)", &EXPORT_SETTINGS.exportCd);
            ImGui::Checkbox("Pressure Drop", &EXPORT_SETTINGS.exportPressureDrop);
            ImGui::Checkbox("L/D Ratio", &EXPORT_SETTINGS.exportLtoDRatio);
        }
        
        ImGui::Separator();
        ImGui::TextWrapped("Last Telemetry Export: Frame %d", EXPORT_SETTINGS.lastExportedTelemetryFrame);
        ImGui::TextWrapped("Last Grid Export: Frame %d (File #%d)", 
                          EXPORT_SETTINGS.lastExportedGridFrame,
                          EXPORT_SETTINGS.gridExportCounter - 1);
    }
    
    ImGui::End();
}


#endif // GRID_VISUALIZER_H