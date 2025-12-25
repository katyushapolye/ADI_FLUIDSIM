/*Made by Raphael Laroca*/
#include <iostream>
#include <time.h>
#include <sstream>

#include "headers/Core/MAC.h"
#include "headers/Core/Functions.h"
#include "headers/Core/Definitions.h"
#include "headers/Core/ConfigReader.h"
#include "headers/Core/GridVisualizer.h"
#include "headers/Solvers/ADI.h"
#include "headers/Solvers/ADI_2D.h"
#include "headers/Solvers/PressureSolver.h"
#include "headers/Solvers/PressureSolver_2D.h"
#include "headers/FLIP.h"


#include "headers/Core/SimulationManager.h"

#include "headers/Utils.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>




//what to do
//continue this, we now only haave on simulation and one mac, so it should make it simpler
//tomorrow, add 3 morre configs with the creation of the simulation manager
// FIX ADI DIFFUSION MATRIX ON THE ADAPTATIVE TIMESTEPPING
//visualization manaage, that handles all the UI stuff (mege grid visualizer into it)

int main(int argc, char *argv[])
{


    SimulationManager::InitializeSimulation();
    SimulationManager::InitializeExportTelemetry();

    
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

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    

    ImGui::StyleColorsDark();
    

    GridVisualizer* visualizer = nullptr;
    visualizer = new GridVisualizer(SIMULATION.GRID_SOL);


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        



        SimulationManager::StepSimulation(visualizer);


        
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



