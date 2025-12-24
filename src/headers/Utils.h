#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <tuple>
#include <string>
#include <amgx_c.h>
#include <ctime>
#include <utility> // for std::forward
#include <time.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>


using Eigen::MatrixXd;
using Eigen::VectorXd;

// Struct definitions
struct CSRMatrix {
    int* row_ptr;
    int* col_ind;
    double* values;
    int num_rows;
    int num_cols;
    int nnz;
};

struct AMGXSolver {
    AMGX_matrix_handle AmgxA;
    AMGX_vector_handle Amgxb, Amgxx;
    AMGX_vector_handle Amgxguess;
    AMGX_solver_handle solver;
    AMGX_resources_handle rsrc;
    AMGX_config_handle config;
};


struct Vec3 {
    double u;
    double v;
    double w;
};

struct Vec2 {
    double u;
    double v;
};


enum class LevelConfiguration {
    LID_CAVITY,
    STEP,
    OBSTACLE,
    DAMBREAK,

};

struct Domain {
    double x0;
    double xf;
    double y0;
    double yf;
    double z0;
    double zf;
};

struct Domain2D {
    double x0;
    double xf;
    double y0;
    double yf;

};


class CPUTimer {
public:
    void start();
    double stop();
private:
    double m_start;  // Changed from std::clock_t
};
// Function declarations
template <typename Derived>
void exportMatrixToFile(const Eigen::MatrixBase<Derived>& matrix, 
                        const std::string& filename, 
                        const std::string& delimiter = ",");

template <typename Derived>
void exportVectorToFile(const Eigen::MatrixBase<Derived>& vector,
                       const std::string& filename,
                       const std::string& delimiter = "\n");

double GetWallTime();
double GetCpuTimeMs();

void writeSparseMatrixToFile(const Eigen::SparseMatrix<double>& matrix, const std::string& filename);

void TDMA(Eigen::MatrixXd& mat, Eigen::VectorXd& font, Eigen::VectorXd& sol);

CSRMatrix* coo_to_csr(int* rows, int* cols, double* values, int nnz, int num_rows, int num_cols);

void free_csr_matrix(CSRMatrix* csr);

void export_csr_to_file(const CSRMatrix* csr, const std::string& filename);

std::string LevelConfigurationToString(LevelConfiguration config);


bool WriteToCSV(double value, std::string levelString, std::string gridSize,std::string dataType);

#endif
