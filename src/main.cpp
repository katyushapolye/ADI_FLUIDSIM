/*Made by Raphael Laroca*/
#include <iostream>
#include <time.h>
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
void InitializeTelemetryExport();

void UpdateTelemetry(int IT,int F_IT,double endTotal,double startTotal,double b4ProjectDiv,int &frame,float simTime);
void DrawSimulationPlots();
void SimulationStep(int& IT, int& frame, double& time, double& b4Project, bool& simulationRunning, GridVisualizer* visualizer);

void RenderUI(int IT, int F_IT, double time, double tF, double frameTime, bool& simulationRunning, bool& stepOnce);
void RenderExportUI();

void CalculateAerodynamicsCoeficients(float simTime);
void DrawAerodynamicsTelemetry();
void HandleExports(int IT, int frame);


//what to do
//tidy tthis main, have 2 more files
//simulation manager that abstract the step with function pointers
//visualization manaage, that handles all the UI stuff (mege grid visualizer into it)

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

    InitializeTelemetryExport();
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    omp_set_num_threads(8);
    
    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(
        mode->width,
        mode->height,
        "ADI Solver Visualization",
        nullptr,
        nullptr
    );

    
    glfwSetWindowPos(window, 0, 0);

    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
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

    //this shos the initial conditions andd solid mask
    if(DIMENSION == 2) {
        SIMULATION2D.GRID_ANT->SetBorder(SIMULATION2D.VelocityBoundaryFunction,SIMULATION2D.PressureBoundaryFunction,0);
        visualizer->UpdateGrid(SIMULATION2D.GRID_ANT);
    }
    else{
        SIMULATION.GRID_ANT->SetBorder(SIMULATION.VelocityBoundaryFunction,SIMULATION.PressureBoundaryFunction,0);
        visualizer->UpdateGrid(SIMULATION.GRID_ANT);
    }
    
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
        RenderExportUI();
        
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
    ImPlot3D::DestroyContext();
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

    if(SIMULATION2D.level == LevelConfiguration::OBSTACLE || SIMULATION.level == LevelConfiguration::OBSTACLE){
            CalculateAerodynamicsCoeficients(time);
    }
    HandleExports(IT, frame);
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

void DrawAerodynamicsTelemetry() {
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
    if(DIMENSION == 2) {
         // Only implemented for 2D
    
    
    /*pressure drop*/
    AERODYNAMICS.time.push_back(simTime);
    //make the points coordinate be selectable with imgui
    //there is informtion about the max range on SIMULATION2D.Nx and SIMULATION2D.Nx

    int maxX = SIMULATION2D.Nx;
    int maxY = SIMULATION2D.Ny;
    float dh = SIMULATION2D.dh; // Get the grid spacing

    static int x1 = SIMULATION2D.Nx / 2;
    static int y1 = SIMULATION2D.Ny / 2;
    static int x2 = SIMULATION2D.Nx / 2 + 1;
    static int y2 = SIMULATION2D.Ny / 2;

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
    double p1 = SIMULATION2D.GRID_SOL->GetP(y1, x1);
    double p2 = SIMULATION2D.GRID_SOL->GetP(y2, x2);
    double pdif = p1 - p2;

    // Display with physical positions
    ImGui::Separator();
    ImGui::Text("Point 1: (%.3f, %.3f) -> P = %.6f", x1 * dh, y1 * dh, p1);
    ImGui::Text("Point 2: (%.3f, %.3f) -> P = %.6f", x2 * dh, y2 * dh, p2);
    ImGui::Text("Pressure Difference: %.6f", pdif);
    ImGui::End();




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
        
        AERODYNAMICS.time.push_back(simTime);
        
        int maxX = SIMULATION.Nx;
        int maxY = SIMULATION.Ny;
        int maxZ = SIMULATION.Nz;

        static int x1 = SIMULATION.Nx / 2;
        static int y1 = SIMULATION.Ny / 2;
        static int z1 = SIMULATION.Nz / 2;
        static int x2 = SIMULATION.Nx / 2 + 1;
        static int y2 = SIMULATION.Ny / 2;
        static int z2 = SIMULATION.Nz / 2;

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






// Add this function to initialize the CSV file with headers
void InitializeTelemetryExport() {
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
        levelStr = LevelConfigurationToString(SIMULATION2D.level);
        exportPath = exportBasePath + "/" + levelStr + "/" +
            std::to_string(SIMULATION2D.GRID_SIZE) + "_2D" + "_re" +
            std::to_string(int(SIMULATION2D.RE)) + "/";
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
        EXPORT_SETTINGS.telemetryFile << ",cfl";
    if (EXPORT_SETTINGS.exportResidual) 
        EXPORT_SETTINGS.telemetryFile << ",residual";
    if (EXPORT_SETTINGS.exportCPUTime) 
        EXPORT_SETTINGS.telemetryFile << ",cpu_time";
    if (EXPORT_SETTINGS.exportGPUTime) 
        EXPORT_SETTINGS.telemetryFile << ",gpu_time";
    
    // Aerodynamics headers (only for 2D with obstacle)
    if (DIMENSION == 2 && SIMULATION2D.level == LevelConfiguration::OBSTACLE) {
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

// Add this function to export telemetry data
void ExportTelemetryData(int iteration) {
    if (!EXPORT_SETTINGS.enabled || !EXPORT_SETTINGS.fileInitialized) return;
    
    EXPORT_SETTINGS.telemetryFile << iteration;
    
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
        if (!TELEMETRY.cfl.empty())
            EXPORT_SETTINGS.telemetryFile << "," << TELEMETRY.cfl.back();
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
    if ( SIMULATION.level == LevelConfiguration::OBSTACLE || SIMULATION2D.level == LevelConfiguration::OBSTACLE) {
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

// Add this new UI function (call it from your main loop)
void RenderExportUI() {
    ImGui::Begin("Export Settings");
    
    bool wasEnabled = EXPORT_SETTINGS.enabled;
    ImGui::Checkbox("Enable Export", &EXPORT_SETTINGS.enabled);
    
    // Auto-initialize file when export is enabled
    if (EXPORT_SETTINGS.enabled && !wasEnabled && !EXPORT_SETTINGS.fileInitialized) {
        InitializeTelemetryExport();
    }
    
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
        if ( SIMULATION.level == LevelConfiguration::OBSTACLE || SIMULATION2D.level == LevelConfiguration::OBSTACLE )  {
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

// Modify the SimulationStep function to include export logic
// Add this at the end of SimulationStep(), right after IT++:
void HandleExports(int IT, int frame) {
    if (!EXPORT_SETTINGS.enabled) return;
    
    // Initialize file on first export (auto-initialize)
    if (!EXPORT_SETTINGS.fileInitialized) {
        InitializeTelemetryExport();
    }
    
    // Export telemetry at telemetry interval
    if ((IT - 1) % EXPORT_SETTINGS.telemetryInterval == 0) {
        ExportTelemetryData(IT);
        EXPORT_SETTINGS.lastExportedTelemetryFrame = IT;
    }
    
    // Export grid at grid interval (if enabled)
    if (EXPORT_SETTINGS.exportGridData && (IT - 1) % EXPORT_SETTINGS.gridInterval == 0) {
        if (DIMENSION == 3) {
            SIMULATION.GRID_SOL->ExportGrid(EXPORT_SETTINGS.gridExportCounter);
        } else if (DIMENSION == 2) {
            SIMULATION2D.GRID_SOL->ExportGrid(EXPORT_SETTINGS.gridExportCounter);
        }
        EXPORT_SETTINGS.lastExportedGridFrame = IT;
        EXPORT_SETTINGS.gridExportCounter++;
    }
}
