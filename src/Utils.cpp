#include "Utils.h"



    

// Non-template function implementations
double GetWallTime() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void CPUTimer::start() {
    m_start = GetWallTime();  // Use wall time
}

double CPUTimer::stop() {
    double end = GetWallTime();  // Use wall time
    return end - m_start;
}



//this has checks for now different systems
void TDMA(Eigen::MatrixXd& mat, Eigen::VectorXd& font, Eigen::VectorXd& sol) {
    const int n = mat.rows();
    
    // Safety exit for empty systems
    if (n <= 0) return;
    
    // Handle 1x1 systems (common at corners/narrow gaps)
    if (n == 1) {
        sol.resize(1);
        if (std::abs(mat(0,0)) > 1e-12) sol(0) = font(0) / mat(0,0);
        else sol(0) = 0.0;
        return;
    }

    sol.resize(n);
    Eigen::VectorXd lower(n-1), main(n), upper(n-1);

    // Extraction with safety bounds
    for(int i = 0; i < n-1; ++i) {
        lower(i) = mat(i+1,i);
        main(i) = mat(i,i);
        upper(i) = mat(i,i+1);
    }
    main(n-1) = mat(n-1,n-1); 

    // Forward elimination
    for(int i = 1; i < n; ++i) {
        if (std::abs(main(i-1)) < 1e-15) continue; // Prevent division by zero
        const double factor = lower(i-1) / main(i-1);
        main(i) -= factor * upper(i-1);
        font(i) -= factor * font(i-1);
    }

    // Backward substitution
    if (std::abs(main(n-1)) > 1e-15) sol(n-1) = font(n-1) / main(n-1);
    else sol(n-1) = 0.0;

    for(int i = n-2; i >= 0; --i) {
        if (std::abs(main(i)) > 1e-15) {
            sol(i) = (font(i) - upper(i) * sol(i+1)) / main(i);
        } else {
            sol(i) = sol(i+1); // Fallback for singular rows
        }
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



std::string LevelConfigurationToString(LevelConfiguration config) {
    switch(config) {
    case LevelConfiguration::LID_CAVITY:
        return "CAVITY";
    case LevelConfiguration::STEP:
        return "STEP";
    case LevelConfiguration::OBSTACLE:
        return "OBSTACLE";
    case LevelConfiguration::DAMBREAK:
        return "DAMBREAK";
    default:
        printf("INVALID LEVEL CONFIG - CHECK LEVEL FUNCTION \n");
        return "EMPTY";
    }
}


