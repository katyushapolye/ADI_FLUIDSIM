############################
# Compilers
############################
CXX := g++

############################
# C++ flags
############################
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
CXXFLAGS += -fopenmp -MMD -MP
CXXFLAGS += -Isrc/headers
CXXFLAGS += -Iinclude

############################
# Eigen
############################
EIGEN_PATH := Eigen
CXXFLAGS += -I$(EIGEN_PATH)

############################
# ImGui / ImPlot / ImPlot3D
############################
IMGUI_DIR := include/imgui-1.92.4
IMPLOT_DIR := include/implot-0.17
IMPLOT3D_DIR := include/implot3d-0.3

CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
CXXFLAGS += -I$(IMPLOT_DIR)
CXXFLAGS += -I$(IMPLOT3D_DIR)

############################
# Libraries (Windows)
############################
############################
# GLFW
############################
GLFW_DIR := $(CURDIR)/libs/glfw-3.4.bin.WIN64
GLFW_LIB := $(GLFW_DIR)/lib-mingw-w64/libglfw3.a

CXXFLAGS += -I$(GLFW_DIR)/include
LDFLAGS  += $(GLFW_LIB)

LDFLAGS :=
LDFLAGS += $(GLFW_LIB)
LDFLAGS += $(ASSIMP_LIB)

# OpenGL + Win32
LDFLAGS += -lopengl32 -lgdi32 -luser32 -lshell32

# Misc
LDFLAGS += -fopenmp -lstdc++ -lpthread

############################
# Directories
############################
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

############################
# Sources
############################
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

IMGUI_SRCS := \
  $(IMGUI_DIR)/imgui.cpp \
  $(IMGUI_DIR)/imgui_demo.cpp \
  $(IMGUI_DIR)/imgui_draw.cpp \
  $(IMGUI_DIR)/imgui_tables.cpp \
  $(IMGUI_DIR)/imgui_widgets.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

IMPLOT_SRCS := \
  $(IMPLOT_DIR)/implot.cpp \
  $(IMPLOT_DIR)/implot_items.cpp \
  $(IMPLOT_DIR)/implot_demo.cpp

IMPLOT3D_SRCS := \
  $(IMPLOT3D_DIR)/implot3d.cpp \
  $(IMPLOT3D_DIR)/implot3d_items.cpp \
  $(IMPLOT3D_DIR)/implot3d_demo.cpp \
  $(IMPLOT3D_DIR)/implot3d_meshes.cpp

############################
# Objects
############################
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp,$(BUILD_DIR)/imgui/%.o,$(IMGUI_SRCS))
IMPLOT_OBJS := $(patsubst $(IMPLOT_DIR)/%.cpp,$(BUILD_DIR)/implot/%.o,$(IMPLOT_SRCS))
IMPLOT3D_OBJS := $(patsubst $(IMPLOT3D_DIR)/%.cpp,$(BUILD_DIR)/implot3d/%.o,$(IMPLOT3D_SRCS))

ALL_OBJS := $(OBJS) $(IMGUI_OBJS) $(IMPLOT_OBJS) $(IMPLOT3D_OBJS)

############################
# Dependencies
############################
DEPS := $(ALL_OBJS:.o=.d)

############################
# Target
############################
TARGET := $(BIN_DIR)/program.exe

all: $(TARGET)

############################
# Directories
############################
MKDIR_P = if not exist "$@" mkdir "$@"

$(BUILD_DIR) \
$(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends \
$(BUILD_DIR)/implot $(BUILD_DIR)/implot3d \
$(BIN_DIR):
	$(MKDIR_P)


############################
# Link
############################
$(TARGET): $(ALL_OBJS) | $(BIN_DIR)
	$(CXX) $(ALL_OBJS) -o $@ $(LDFLAGS)

############################
# Compile rules
############################
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/imgui/%.o: $(IMGUI_DIR)/%.cpp | $(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/imgui/backends/%.o: $(IMGUI_DIR)/backends/%.cpp | $(BUILD_DIR)/imgui/backends
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/implot/%.o: $(IMPLOT_DIR)/%.cpp | $(BUILD_DIR)/implot
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/implot3d/%.o: $(IMPLOT3D_DIR)/%.cpp | $(BUILD_DIR)/implot3d
	$(CXX) $(CXXFLAGS) -c $< -o $@

############################
# Run
############################
run: $(TARGET)
	./$(TARGET)

############################
# Clean
############################
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all run clean
