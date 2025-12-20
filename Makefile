############################
# AMGX paths
############################
AMGX_PATH := $(HOME)/SDK/AMGX
AMGX_INCLUDE_PATH := $(AMGX_PATH)/include
AMGX_BUILD_PATH := $(AMGX_PATH)/build

############################
# CUDA paths
############################
CUDA_PATH := /usr/local/cuda
CUDA_LIB_PATH := $(CUDA_PATH)/lib64

############################
# Eigen
############################
EIGEN_PATH := Eigen

############################
# ImGui / ImPlot / ImPlot3D
############################
IMGUI_DIR := include/imgui-1.92.4
IMPLOT_DIR := include/implot-0.17
IMPLOT3D_DIR := include/implot3d-0.3

############################
# Compilers
############################
CXX := g++
NVCC := nvcc

############################
# C++ flags
############################
CXXFLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy
CXXFLAGS += -O3 -fopenmp -MMD -MP
CXXFLAGS += -Isrc/headers
CXXFLAGS += -I$(CUDA_PATH)/include
CXXFLAGS += -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
CXXFLAGS += -I$(EIGEN_PATH)
CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
CXXFLAGS += -I$(IMPLOT_DIR)
CXXFLAGS += -I$(IMPLOT3D_DIR)

############################
# NVCC flags
############################
NVCCFLAGS := -std=c++17 -O3 -Xcompiler -Wall,-Wextra -MMD -MP
NVCCFLAGS += -Isrc/headers
NVCCFLAGS += -I$(CUDA_PATH)/include
NVCCFLAGS += -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
NVCCFLAGS += -I$(EIGEN_PATH)

############################
# Linker flags
############################
LDFLAGS := -L$(CUDA_LIB_PATH) -L$(AMGX_BUILD_PATH)
LDFLAGS += -lamgx -lamgxsh -lmpi
LDFLAGS += -lcudart -lcublas -lcusparse -lcusolver
LDFLAGS += -lcudadevrt -lcudart_static
LDFLAGS += -fopenmp -lrt -lpthread -ldl
LDFLAGS += -lglfw -lGL

############################
# Directories
############################
SRC_DIR := src
CUDA_SRC_DIR := $(SRC_DIR)/cuda
BUILD_DIR := build
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
BIN_DIR := bin

############################
# Sources
############################
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)

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
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_BUILD_DIR)/%.o,$(CUDA_SRCS))

IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp,$(BUILD_DIR)/imgui/%.o,$(IMGUI_SRCS))
IMPLOT_OBJS := $(patsubst $(IMPLOT_DIR)/%.cpp,$(BUILD_DIR)/implot/%.o,$(IMPLOT_SRCS))
IMPLOT3D_OBJS := $(patsubst $(IMPLOT3D_DIR)/%.cpp,$(BUILD_DIR)/implot3d/%.o,$(IMPLOT3D_SRCS))

ALL_OBJS := $(OBJS) $(CUDA_OBJS) $(IMGUI_OBJS) $(IMPLOT_OBJS) $(IMPLOT3D_OBJS)

############################
# Dependencies
############################
DEPS := $(ALL_OBJS:.o=.d)

############################
# Target
############################
TARGET := $(BIN_DIR)/program

all: $(TARGET)

############################
# Directories
############################
$(BUILD_DIR) $(CUDA_BUILD_DIR) \
$(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends \
$(BUILD_DIR)/implot $(BUILD_DIR)/implot3d \
$(BIN_DIR):
	mkdir -p $@

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

$(CUDA_BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

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
	OMP_NUM_THREADS=8 ./$(TARGET)

############################
# Clean
############################
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all run clean

