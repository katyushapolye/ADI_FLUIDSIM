/*Made by Raphael Laroca*/
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <sstream>

#include "headers/MAC.h"
#include "headers/ADI.h"
#include "headers/MAC_2D.h"
#include "headers/ADI_2D.h"
#include "headers/PressureSolver.h"
#include "headers/PressureSolver_2D.h"
#include "headers/Functions.h"
#include "headers/Definitions.h"
#include "headers/ConfigReader.h"
#include "headers/GridVisualizer.h"
#include "headers/Utils.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

void InitializeSimulation(const std::string& configFile = "simulation_config.txt");
void UpdateTelemetry(int IT,int F_IT,double endTotal,double startTotal,double b4ProjectDiv,int &frame,float simTime);
void DrawSimulationPlots();
void SimulationStep(int& IT, int& frame, double& time, double& b4Project, bool& simulationRunning, GridVisualizer* visualizer);
void RenderUI(int IT, int F_IT, double time, double tF, double frameTime, bool& simulationRunning, bool& stepOnce);
void CalculateAerodynamicsCoeficients(float simTime);
void DrawAerodynamicsTelemetry();

int main(int argc, char *argv[])
{
    // Initialize simulation
    if(argc == 2){
        std::string inputFile(argv[1]);
        InitializeSimulation(inputFile);
    }
    else{
        InitializeSimulation();
    }
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1600, 900, "ADI Solver Visualization", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Setup style
    ImGui::StyleColorsDark();
    
    // Create visualizer based on dimension
    GridVisualizer* visualizer = nullptr;
    if(DIMENSION == 3) {
        visualizer = new GridVisualizer(SIMULATION.GRID_SOL);
    } else if(DIMENSION == 2) {
        visualizer = new GridVisualizer(SIMULATION2D.GRID_SOL);
    }
    
    // Simulation variables
    int IT = 1;
    int frame = 1;
    double time = 0.0;
    double tF = 100000.0;
    
    // Use correct dt based on dimension
    double dt = (DIMENSION == 3) ? SIMULATION.dt : SIMULATION2D.dt;
    int F_IT = tF / dt;
    
    double startTotal = 0.0, endTotal = 0.0;
    double b4Project = 0.0;
    
    bool simulationRunning = false;
    bool stepOnce = false;
    
    // Main loop
    while (!glfwWindowShouldClose(window) && time < tF)
    {
        glfwPollEvents();
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Render UI
        double frameTime = endTotal - startTotal;
        RenderUI(IT, F_IT, time, tF, frameTime, simulationRunning, stepOnce);
        
        // Draw simulation plots
        DrawSimulationPlots();
        DrawAerodynamicsTelemetry();
        
        // Run simulation step if needed
        if (simulationRunning || stepOnce) {
            startTotal = GetWallTime();
            SimulationStep(IT, frame, time, b4Project, simulationRunning, visualizer);
            endTotal = GetWallTime();
            stepOnce = false;
        }
        
        // Render visualizer (only for 3D)
        if(visualizer != nullptr) {
            visualizer->Render();
        }
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    if(visualizer != nullptr) {
        delete visualizer;
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}

void InitializeSimulation(const std::string& configFile) {
    ConfigReader::loadConfig(SIMULATION, configFile);
    
    if(DIMENSION == 3){
        SIMULATION.GRID_ANT->ExportGrid(0);
        TELEMETRY.Push(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        
        std::cout << "3D MAC Grid initialized - Parameters\n"
                  << "-dt = " << std::to_string(SIMULATION.dt) << "\n"
                  << "-dh = " << std::to_string(SIMULATION.dh) 
                  << "\n-Nx = " << std::to_string(SIMULATION.Nx) 
                  << "\n-Ny = " << std::to_string(SIMULATION.Ny)
                  << "\n-Nz = " << std::to_string(SIMULATION.Nz) << std::endl;
        std::cout << "Total node count: " 
                  << (SIMULATION.Nx * SIMULATION.Ny * SIMULATION.Nz) * 3 
                  << " - Re = " << SIMULATION.RE 
                  << " - Grid size: " << SIMULATION.GRID_SIZE 
                  << " - Tolerance: " << SIMULATION.TOLERANCE << std::endl;
        
        ADI::InitializeADI(SIMULATION.GRID_SOL, SIMULATION.dt, 
                          SIMULATION.VelocityBoundaryFunction, ZERO, 
                          SIMULATION.PressureBoundaryFunction);
        PressureSolver::InitializePressureSolver(SIMULATION.GRID_SOL, SIMULATION.dt);
    }
    else if(DIMENSION == 2){
        SIMULATION2D.GRID_ANT->ExportGrid(0);
        TELEMETRY.Push(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        
        std::cout << "2D MAC Grid initialized - Parameters\n"
                  << "-dt = " << std::to_string(SIMULATION2D.dt) << "\n"
                  << "-dh = " << std::to_string(SIMULATION2D.dh) 
                  << "\n-Nx = " << std::to_string(SIMULATION2D.Nx) 
                  << "\n-Ny = " << std::to_string(SIMULATION2D.Ny) << "\n";
        std::cout << "Total node count: " 
                  << (SIMULATION2D.Nx * SIMULATION2D.Ny) * 2
                  << " - Re = " << SIMULATION2D.RE 
                  << " - Grid size: " << SIMULATION2D.GRID_SIZE 
                  << " - Tolerance: " << SIMULATION2D.TOLERANCE << std::endl;
        
        ADI2D::InitializeADI2D(SIMULATION2D.GRID_SOL, SIMULATION2D.dt, 
                               SIMULATION2D.VelocityBoundaryFunction, ZERO2D, 
                               SIMULATION2D.PressureBoundaryFunction);

        PressureSolver2D::InitializePressureSolver(SIMULATION2D.GRID_SOL, SIMULATION2D.dt);
    }
}

void SimulationStep(int& IT, int& frame, double& time, double& b4Project, bool& simulationRunning, GridVisualizer* visualizer) {
    if(DIMENSION == 3){
        // Diffusion + Convection
        ADI::SolveADIStep(SIMULATION.GRID_ANT, SIMULATION.GRID_SOL, time);
        
        // Pressure Projection
        if(SIMULATION.level == LevelConfiguration::STEP || 
           SIMULATION.level == LevelConfiguration::OBSTACLE) {
            SIMULATION.GRID_SOL->SetNeumannBorder();
        }
        
        PressureSolver::SolvePressure_AMGX(SIMULATION.GRID_SOL);
        b4Project = SIMULATION.GRID_SOL->GetDivSum();
        PressureSolver::ProjectPressure(SIMULATION.GRID_SOL);
        
        time += SIMULATION.dt;
        
        double startTotal = GetWallTime();
        double endTotal = startTotal;
        int F_IT = 100000.0 / SIMULATION.dt;
        
        UpdateTelemetry(IT, F_IT, endTotal, startTotal, b4Project, frame, time);

        if (SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL) < SIMULATION.TOLERANCE) {
            if(SIMULATION.level == LevelConfiguration::STEP || 
               SIMULATION.level == LevelConfiguration::OBSTACLE) {
                SIMULATION.GRID_SOL->SetNeumannBorder();
            }
            SIMULATION.GRID_SOL->ExportGrid(frame);
            frame++;
            simulationRunning = false; 
        }
        
        SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);
        IT++;
        
        // Update visualizer with new grid data
        if(visualizer != nullptr) {
            visualizer->UpdateGrid(SIMULATION.GRID_SOL);
        }
    }
    else if(DIMENSION == 2){
        // Diffusion + Convection
        ADI2D::SolveADIStep(SIMULATION2D.GRID_ANT, SIMULATION2D.GRID_SOL, time);
        
        // Pressure Projection
        if(SIMULATION2D.level == LevelConfiguration::STEP || 
           SIMULATION2D.level == LevelConfiguration::OBSTACLE) {
            SIMULATION2D.GRID_SOL->SetNeumannBorder();
        }


        PressureSolver2D::SolvePressure_AMGX(SIMULATION2D.GRID_SOL);
        b4Project = SIMULATION2D.GRID_SOL->GetDivSum();
        PressureSolver2D::ProjectPressure(SIMULATION2D.GRID_SOL);

        time += SIMULATION2D.dt;

        double startTotal = GetWallTime();
        double endTotal = startTotal;
        int F_IT = 100000.0 / SIMULATION2D.dt;

        UpdateTelemetry(IT, F_IT, endTotal, startTotal, b4Project, frame, time);
        if(SIMULATION2D.level == LevelConfiguration::OBSTACLE){
            CalculateAerodynamicsCoeficients(time);
        }

        if (IT > 10 && SIMULATION2D.GRID_ANT->MaxAbsoluteDifference(*SIMULATION2D.GRID_SOL) < SIMULATION2D.TOLERANCE) {
            if(SIMULATION2D.level == LevelConfiguration::STEP || 
               SIMULATION2D.level == LevelConfiguration::OBSTACLE) {
                SIMULATION2D.GRID_SOL->SetNeumannBorder();
            }
            SIMULATION2D.GRID_SOL->ExportGrid(frame);
            frame++;
            simulationRunning = false; 
        }

        SIMULATION2D.GRID_ANT->CopyGrid(*SIMULATION2D.GRID_SOL);
        IT++;

        // Update visualizer with new grid data (2D version)
        if(visualizer != nullptr) {
            visualizer->UpdateGrid(SIMULATION2D.GRID_SOL);
        }

    }
}

void RenderUI(int IT, int F_IT, double time, double tF, double frameTime, bool& simulationRunning, bool& stepOnce) {
    ImGui::Begin("Simulation Control");
    
    ImGui::Text("Iteration: %d / %d", IT, F_IT);
    ImGui::Text("Time: %.4f / %.4f", time, tF);
    
    // Display divergence based on dimension
    if(DIMENSION == 3) {
        ImGui::Text("Divergence: %.6e", SIMULATION.GRID_SOL->GetDivSum());
    } else if(DIMENSION == 2) {
        ImGui::Text("Divergence: %.6e", SIMULATION2D.GRID_SOL->GetDivSum());
    }
    
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

void DrawSimulationPlots()
{
    ImGui::Begin("Simulation Telemetry");
    
    float avail = ImGui::GetContentRegionAvail().x;
    float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    ImVec2 plot_size(plot_w, 300);
    
    // ---- Row 1 ----
    if (ImPlot::BeginPlot("Divergence Evolution", plot_size)) {
        ImPlot::SetupAxes("Time", "Divergence Sum",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        ImPlot::PlotLine("After Projection",
            TELEMETRY.time.data(), TELEMETRY.div_sum.data(), TELEMETRY.time.size());
        ImPlot::PlotLine("Before Projection",
            TELEMETRY.time.data(), TELEMETRY.div_sum_before_proj.data(), TELEMETRY.time.size());
        ImPlot::EndPlot();
    }
    
    ImGui::SameLine();
    
    if (ImPlot::BeginPlot("CFL Condition", plot_size)) {
        ImPlot::SetupAxes("Time", "CFL");
        ImPlot::PlotLine("CFL",
            TELEMETRY.time.data(), TELEMETRY.cfl.data(), TELEMETRY.time.size());
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

        const double t = TELEMETRY.time.back();
        const double window = 5.0;

        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t-window*0.1,
            t+window,
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
        y_max = std::max(y_max, 1.2 * std::max(values[0], values[1]));
    
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

void DrawAerodynamicsTelemetry() {
    ImGui::Begin("Aerodynamics Telemetry");
    
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
            
            int start = std::max(0, (int)data.size() - n_samples);
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
            int n_samples = std::min(100, (int)AERODYNAMICS.Cl.size());
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


void UpdateTelemetry(int IT,int F_IT,double endTotal,double startTotal,double b4ProjectDiv,int &frame,float simTime){
    if(DIMENSION == 3){
        double cfl = (SIMULATION.GRID_SOL->GetMaxVelocity() / SIMULATION.dh) * SIMULATION.dt;
        double residual = SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL);

        TELEMETRY.Push(
            simTime, 
            SIMULATION.GRID_SOL->GetDivSum(),
            b4ProjectDiv,
            cfl,
            residual,
            SIMULATION.lastADISolveTime,
            SIMULATION.lastPressureSolveTime
        );
    }
    else if(DIMENSION == 2){
        double cfl = (SIMULATION2D.GRID_SOL->GetMaxVelocity() / SIMULATION2D.dh) * SIMULATION2D.dt;
        double residual = SIMULATION2D.GRID_ANT->MaxAbsoluteDifference(*SIMULATION2D.GRID_SOL);

        TELEMETRY.Push(
            simTime, 
            SIMULATION2D.GRID_SOL->GetDivSum(),
            b4ProjectDiv,
            cfl,
            residual,
            SIMULATION2D.lastADISolveTime,
            SIMULATION2D.lastPressureSolveTime
        );
    }
}

void CalculateAerodynamicsCoeficients(float simTime) {
    
    /*pressure drop*/
    AERODYNAMICS.time.push_back(simTime);
    //make the points coordinate be selectable with imgui
    //there is informtion about the max range on SIMULATION2D.Nx and SIMULATION2D.Nx

    int maxX = SIMULATION2D.Nx;
    int maxY = SIMULATION2D.Ny;

    static int x1 = 125 / 2;
    static int y1 = 93  / 2;
    static int x2 = 125 / 2;
    static int y2 = 157 / 2;

    ImGui::Begin("Pressure Difference");

    // Sliders (safe & easy)
    ImGui::SliderInt("X1", &x1, 0, maxX - 1);
    ImGui::SliderInt("Y1", &y1, 0, maxY - 1);
    ImGui::Separator();
    ImGui::SliderInt("X2", &x2, 0, maxX - 1);
    ImGui::SliderInt("Y2", &y2, 0, maxY - 1);

    // Compute difference
    double p1 = SIMULATION2D.GRID_SOL->GetP(y1, x1);
    double p2 = SIMULATION2D.GRID_SOL->GetP(y2, x2);
    double pdif = p1 - p2;

    ImGui::Text("P1 = %.6f", p1);
    ImGui::Text("P2 = %.6f", p2);
    ImGui::Text("Difference = %.6f", pdif);

    ImGui::End();




    double dh = SIMULATION2D.dh;
    double mi = SIMULATION2D.EPS; // kinematic viscosity
    MAC2D* grid = SIMULATION2D.GRID_SOL;

    double dudx = 0.0;
    double dudy = 0.0;
    double dvdx = 0.0;
    double dvdy = 0.0;

    double Fd = 0.0;
    double Fl = 0.0;
    
    // For all non-boundary cell centers
    for(int i = 1; i < SIMULATION2D.Ny-1; i++) {
        for(int j = 1; j < SIMULATION2D.Nx-1; j++) {
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

    //the coefficient is defined as 2/(Umean^2 L)
    double Cd = (2.0/(1.0*0.1))*Fd;
    double Cl = (2.0/(1.0*0.1))*Fl;


    
    // Store in telemetry for visualization
    AERODYNAMICS.Cl.push_back(Cl);
    AERODYNAMICS.Cd.push_back(Cd);
    AERODYNAMICS.pressure_drop.push_back(pdif);
    AERODYNAMICS.LtoDRatio.push_back(Cl/Cd);
}