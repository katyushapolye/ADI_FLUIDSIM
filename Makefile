# AMGX paths
AMGX_PATH := $(HOME)/SDK/AMGX
AMGX_INCLUDE_PATH := $(AMGX_PATH)/include
AMGX_BUILD_PATH := $(AMGX_PATH)/build

# CUDA paths
CUDA_PATH := /usr/local/cuda
CUDA_LIB_PATH := $(CUDA_PATH)/lib64

# Eigen path
EIGEN_PATH := Eigen

# Compilers and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Isrc/headers -fopenmp -O3 -MMD -MP -Wno-unused-parameter -Wno-deprecated-copy
CXXFLAGS += -I$(CUDA_PATH)/include
CXXFLAGS += -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
CXXFLAGS += -I$(EIGEN_PATH)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O3 -Isrc/headers -Xcompiler -Wall,-Wextra -MMD -MP
NVCCFLAGS += -I$(CUDA_PATH)/include
NVCCFLAGS += -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
NVCCFLAGS += -I$(EIGEN_PATH)

# The order matters! Libraries should come after the objects that depend on them
LDFLAGS := -L$(CUDA_LIB_PATH) -L$(AMGX_BUILD_PATH) \
 -lamgx -lamgxsh -lmpi \
 -lcudart -lcublas -lcusparse -lcusolver \
 -lcudadevrt -lcudart_static \
 -fopenmp -lrt -lpthread -ldl

# Directories
SRC_DIR := src
CUDA_SRC_DIR := $(SRC_DIR)/cuda
HEADER_DIR := src/headers
BUILD_DIR := build
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
BIN_DIR := bin

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_BUILD_DIR)/%.o,$(CUDA_SRCS))

# Dependency files
DEPS := $(OBJS:.o=.d) $(CUDA_OBJS:.o=.d)

# Executable
TARGET := $(BIN_DIR)/program

# Default target
all: $(TARGET)

# Create directories
$(BUILD_DIR) $(CUDA_BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Link executable
$(TARGET): $(OBJS) $(CUDA_OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) $(CUDA_OBJS) -o $@ $(LDFLAGS)

# Compile C++ sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
$(CUDA_BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Include dependency files
-include $(DEPS)

.PHONY: all run clean