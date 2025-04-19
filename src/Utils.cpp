#include "Utils.h"
#include <time.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <amgx_c.h>
#include "Definitions.h"
#include <sstream>
#include <iomanip>
#include <filesystem>



void CPUTimer::start() {
            m_start = std::clock();
        }
        
        double CPUTimer::stop() {
            std::clock_t end = std::clock();
            return static_cast<double>(end - m_start) / CLOCKS_PER_SEC;
        }
    




// Template function implementations
template <typename Derived>
void exportMatrixToFile(const Eigen::MatrixBase<Derived>& matrix, 
                       const std::string& filename, 
                       const std::string& delimiter) {
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write matrix data
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            outFile << matrix(i, j);
            if (j < matrix.cols() - 1) {
                outFile << delimiter;
            }
        }
        outFile << "\n";
    }
    
    outFile.close();
    std::cout << "Matrix successfully exported to " << filename << std::endl;
}

template <typename Derived>
void exportVectorToFile(const Eigen::MatrixBase<Derived>& vector,
                        const std::string& filename,
                        const std::string& delimiter) {
    static_assert(Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1,
                 "exportVectorToFile: Input must be a vector (either row or column)");
    
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write vector data
    for (int i = 0; i < vector.size(); ++i) {
        outFile << vector(i);
        if (i < vector.size() - 1) {
            outFile << delimiter;
        }
    }
    
    outFile.close();
    std::cout << "Vector successfully exported to " << filename << std::endl;
}

// Explicit template instantiations
// You would need to add instantiations for all the types you use with these templates
template void exportMatrixToFile<Eigen::MatrixXd>(const Eigen::MatrixBase<Eigen::MatrixXd>& matrix, 
                                                const std::string& filename, 
                                                const std::string& delimiter);
template void exportVectorToFile<Eigen::VectorXd>(const Eigen::MatrixBase<Eigen::VectorXd>& vector,
                                                const std::string& filename,
                                                const std::string& delimiter);

// Non-template function implementations
double GetWallTime() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void writeSparseMatrixToFile(const Eigen::SparseMatrix<double>& matrix, const std::string& filename) {
    // Open file in binary mode
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write matrix dimensions (rows, cols)
    int rows = matrix.rows();
    int cols = matrix.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));

    // Write number of non-zero elements
    int nnz = matrix.nonZeros();
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(int));

    // Write the data in COO format (row, col, value)
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            double value = it.value();
            
            out.write(reinterpret_cast<const char*>(&row), sizeof(int));
            out.write(reinterpret_cast<const char*>(&col), sizeof(int));
            out.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }

    out.close();
}

void TDMA(Eigen::MatrixXd& mat, Eigen::VectorXd& font, Eigen::VectorXd& sol) {
    const int n = mat.rows();
    sol.resize(n);
    
    Eigen::VectorXd lower(n-1), main(n), upper(n-1);
    for(int i = 0; i < n-1; ++i) {
        lower(i) = mat(i+1,i);    // Subdiagonal
        main(i) = mat(i,i);       // Main diagonal
        upper(i) = mat(i,i+1);    // Superdiagonal
    }
    main(n-1) = mat(n-1,n-1);     // Last main diagonal element

    // Forward elimination
    for(int i = 1; i < n; ++i) {
        const double factor = lower(i-1) / main(i-1);
        main(i) -= factor * upper(i-1);
        font(i) -= factor * font(i-1);
    }

    // Backward substitution
    sol(n-1) = font(n-1) / main(n-1);
    for(int i = n-2; i >= 0; --i) {
        sol(i) = (font(i) - upper(i) * sol(i+1)) / main(i);
    }
}

CSRMatrix* coo_to_csr(int* rows, int* cols, double* values, int nnz, int num_rows, int num_cols) {
    // Allocate the CSR matrix structure on heap
    CSRMatrix* csr = new CSRMatrix;
    csr->num_rows = num_rows;
    csr->num_cols = num_cols;
    csr->nnz = nnz;
    
    // Create a vector of tuples (row, col, value) for sorting
    std::vector<std::tuple<int, int, double>> coo_entries;
    coo_entries.reserve(nnz);
    
    for (int i = 0; i < nnz; ++i) {
        coo_entries.emplace_back(rows[i], cols[i], values[i]);
    }
    
    // Sort by row, then by column within each row
    std::sort(coo_entries.begin(), coo_entries.end(),
        [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) < std::get<0>(b);
            }
            return std::get<1>(a) < std::get<1>(b);
        });
    
    // Allocate memory for CSR arrays
    csr->row_ptr = new int[num_rows + 1];
    csr->col_ind = new int[nnz];
    csr->values = new double[nnz];
    
    // Initialize row_ptr with zeros
    for (int i = 0; i <= num_rows; ++i) {
        csr->row_ptr[i] = 0;
    }
    
    // Process sorted entries to fill col_ind and values
    int current_row = -1;
    int entry_index = 0;
    for (const auto& entry : coo_entries) {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);
        double val = std::get<2>(entry);
        
        // Count non-zeros per row
        while (current_row < row) {
            current_row++;
            csr->row_ptr[current_row + 1] = csr->row_ptr[current_row];
        }
        
        csr->col_ind[entry_index] = col;
        csr->values[entry_index] = val;
        csr->row_ptr[row + 1]++;
        entry_index++;
    }
    
    // Fill remaining row pointers if there are empty rows at the end
    while (current_row < num_rows - 1) {
        current_row++;
        csr->row_ptr[current_row + 1] = csr->row_ptr[current_row];
    }
    
    return csr;
}

void free_csr_matrix(CSRMatrix* csr) {
    if (csr) {
        delete[] csr->row_ptr;
        delete[] csr->col_ind;
        delete[] csr->values;
        delete csr;
    }
}

void export_csr_to_file(const CSRMatrix* csr, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    // Write header information
    outfile << "CSR Matrix Format\n";
    outfile << "Rows: " << csr->num_rows << "\n";
    outfile << "Columns: " << csr->num_cols << "\n";
    outfile << "Non-zero entries: " << csr->nnz << "\n\n";
    
    // Write row pointers
    outfile << "Row pointers (row_ptr):\n";
    for (int i = 0; i <= csr->num_rows; ++i) {
        outfile << csr->row_ptr[i] << " ";
    }
    outfile << "\n\n";
    
    // Write column indices
    outfile << "Column indices (col_ind):\n";
    for (int i = 0; i < csr->nnz; ++i) {
        outfile << csr->col_ind[i] << " ";
    }
    outfile << "\n\n";
    
    // Write values
    outfile << "Values:\n";
    for (int i = 0; i < csr->nnz; ++i) {
        outfile << csr->values[i] << " ";
    }
    outfile << "\n";
    
    outfile.close();
    std::cout << "CSR matrix successfully exported to " << filename << "\n";
}

std::string LevelConfigurationToString(LevelConfiguration config) {
    if(config == LevelConfiguration::LID_CAVITY) {
        return "CAVITY";
    }
    if(config == LevelConfiguration::STEP) {
        return "STEP";
    }
    if(config == LevelConfiguration::OBSTACLE) {
        return "OBSTACLE";
    }
    else {
        printf("INVALID LEVEL CONFIG - CHECK LEVEL FUNCTION \n");
        return "EMPTY";
    }
}

void PrintAMGXMatrix(AMGXSolver* AMGX_Handle, int MatSize) {
    // First, just print some basic info about the matrix without downloading it
    std::cout << "AMGX Matrix Info:" << std::endl;
    std::cout << "Matrix size: " << MatSize << " x " << MatSize << std::endl;
    
    int nnz;
    AMGX_matrix_get_nnz(AMGX_Handle->AmgxA, &nnz);
    std::cout << "Non-zeros reported by AMGX: " << nnz << std::endl;
    
    // Add diagnostic info for solver
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(AMGX_Handle->solver, &status);
    std::cout << "Solver status: " << status << std::endl;
    
    // Try to get some solver statistics if possible
    int iterations = 0;
    double residual = 0.0;
    AMGX_solver_get_iterations_number(AMGX_Handle->solver, &iterations);
    AMGX_solver_get_iteration_residual(AMGX_Handle->solver, 0, 0, &residual);
    std::cout << "Iterations: " << iterations << ", Last residual: " << residual << std::endl;
}

bool WriteToCSV(double value, std::string levelString, std::string gridSize,std::string dataType) {
    try {
        std::string exportBasePath = "Data/" + dataType + "/";
        std::ostringstream oss;
        oss << exportBasePath <<  dataType  << "_" << levelString << "_" << gridSize << "_re" << SIMULATION.RE << ".csv";
        std::string filename = oss.str();
        std::filesystem::create_directories(exportBasePath);
        // Open file in append mode
        std::ofstream file(filename, std::ios::app);
        
        // Check if file opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }
        
        // Write the value followed by a comma and newline
        file << value << "," << std::endl;
        
        // Close the file
        file.close();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return false;
    }
}