#include "headers/Solvers/DiffusionSolver2D.h"

// Static member initialization for DiffusionSolver2D

// Grid dimensions
int DiffusionSolver2D::Nx = 0;
int DiffusionSolver2D::Ny = 0;

// Non-zero count
int DiffusionSolver2D::NON_ZERO = 0;

// Time step
double DiffusionSolver2D::dt = 0.0;

// COO format arrays for U
int* DiffusionSolver2D::u_collums = nullptr;
int* DiffusionSolver2D::u_rows = nullptr;
double* DiffusionSolver2D::u_values = nullptr;

// COO format arrays for V
int* DiffusionSolver2D::v_collums = nullptr;
int* DiffusionSolver2D::v_rows = nullptr;
double* DiffusionSolver2D::v_values = nullptr;

// Sparse matrices
SparseMatrix DiffusionSolver2D::U_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver2D::V_DIFFUSION_MATRIX_EIGEN;

// CSR matrices
CSRMatrix* DiffusionSolver2D::U_DIFFUSION_MATRIX = nullptr;
CSRMatrix* DiffusionSolver2D::V_DIFFUSION_MATRIX = nullptr;

// AMGX solver
AMGXSolver* DiffusionSolver2D::AMGX_Handle = nullptr;

// Index mapping vectors
VectorXd DiffusionSolver2D::U_IDP;
VectorXd DiffusionSolver2D::V_IDP;

// Mask vectors
std::vector<Tensor<double,2>> DiffusionSolver2D::U_MASK;
std::vector<Tensor<double,2>> DiffusionSolver2D::V_MASK;

// Eigen solver
Eigen::BiCGSTAB<SparseMatrix>* DiffusionSolver2D::solver = nullptr;

void DiffusionSolver2D::InitializeDiffusionSolver(MAC* grid){

    DiffusionSolver2D::Nx = SIMULATION.Nx;
    DiffusionSolver2D::Ny = SIMULATION.Ny;
    DiffusionSolver2D::U_MASK.reserve(grid->GetFluidCellCount()+1);
    DiffusionSolver2D::V_MASK.reserve(grid->GetFluidCellCount()+1);
    DiffusionSolver2D::dt = SIMULATION.dt;
    double dh = SIMULATION.dh;
    DiffusionSolver2D::U_IDP = VectorXd((Nx+1) * Ny);
    DiffusionSolver2D::U_IDP.setConstant(-1);  
    DiffusionSolver2D::V_IDP = VectorXd((Nx) * (Ny+1));
    DiffusionSolver2D::V_IDP.setConstant(-1);  


}





void DiffusionSolver2D::Update_V_DiffusionMatrix(MAC* grid){
    double dh = SIMULATION.dh;


    DiffusionSolver2D::V_IDP = VectorXd((Nx) * (Ny+1));
    DiffusionSolver2D::V_IDP.setConstant(-1);  
    int c = 0;
    V_MASK.clear();
    // ========== V COMPONENT ==========
    
    // First pass: count DOFs and build masks
    int v_dof_count = 0;
    for (int i = 0; i < Ny+1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            Tensor<double, 2> mask = Tensor<double, 2>(3, 3);
            mask.setZero();
            
            // Boundary nodes
            if (i == 0 || j == 0 || i==1 || i == Ny || i==Ny-1 || j == Nx)
            {
                DiffusionSolver2D::V_MASK.push_back(mask);
                Set_V_IDP(i, j, -1);
                continue;
            }
            
            // Non-fluid face check
            if (grid->GetSolid(i, j) != FLUID_CELL && grid->GetSolid(i-1, j) != FLUID_CELL){
                DiffusionSolver2D::V_MASK.push_back(mask);
                Set_V_IDP(i, j, -1);
                continue;
            }

            // This is a valid DOF - build stencil
            //i dont know why
            //i dont want to know why
            // this feels wrong but it works perfeclty, for some reason i check 
            //for +1, but apply on -1, it works on borders
            if(grid->GetSolid(i-1, j+1) == FLUID_CELL || grid->GetSolid(i, j+1) == FLUID_CELL){
                mask(1, 2) = 1.0;
                mask(1, 1) -= 1.0;
            }

            else if(grid->GetSolid(i-1, j-1) == SOLID_CELL || grid->GetSolid(i, j-1) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            if(grid->GetSolid(i-1, j-1) == FLUID_CELL || grid->GetSolid(i, j-1) == FLUID_CELL){
                mask(1, 0) = 1.0;
                mask(1, 1) -= 1.0;
            }

                // Dirichlet conditions
            else if(grid->GetSolid(i-1, j+1) == SOLID_CELL || grid->GetSolid(i, j+1) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            if(grid->GetSolid(i, j) == FLUID_CELL){
                mask(2, 1) = 1.0;
                mask(1, 1) -= 1.0;
            }

            if(grid->GetSolid(i-1, j) == FLUID_CELL){
                mask(0, 1) = 1.0;
                mask(1, 1) -= 1.0;
            }

            //

            if(grid->GetSolid(i, j) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            if(grid->GetSolid(i-1, j) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            Set_V_IDP(i, j, v_dof_count);
            v_dof_count++;
            DiffusionSolver2D::V_MASK.push_back(mask);
        }
    }

    // Second pass: count non-zeros
    int v_nnz = 0;
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            int MAT_LINE = Get_V_IDP(i, j);
            if (MAT_LINE == -1) continue;
            
            Tensor<double, 2> mask = GetVMask(i, j);
            
            if (mask(1, 1) != 0.0) v_nnz++;
            if (mask(2, 1) != 0.0 && Get_V_IDP(i + 1, j) != -1) v_nnz++;
            if (mask(0, 1) != 0.0 && Get_V_IDP(i - 1, j) != -1) v_nnz++;
            if (mask(1, 2) != 0.0 && Get_V_IDP(i, j + 1) != -1) v_nnz++;
            if (mask(1, 0) != 0.0 && Get_V_IDP(i, j - 1) != -1) v_nnz++;
        }
    }

    std::cout << "V component: DOFs = " << v_dof_count << ", NNZ = " << v_nnz << std::endl;

    // Allocate arrays
    v_collums = (int *)malloc(sizeof(int) * v_nnz);
    v_rows = (int *)malloc(sizeof(int) * v_nnz);
    v_values = (double *)calloc(v_nnz, sizeof(double));

    // Third pass: assemble matrix
    c = 0;
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            Tensor<double, 2> mask = GetVMask(i, j);
            int MAT_LINE = Get_V_IDP(i, j);
            
            if (MAT_LINE == -1) continue;

            // Diagonal
            if (mask(1, 1) != 0.0)
            {
                v_rows[c] = MAT_LINE;
                v_collums[c] = MAT_LINE;
                v_values[c] = mask(1, 1);
                c++;
            }

            // +Y neighbor
            if (mask(2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i + 1, j);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = mask(2, 1);
                    c++;
                }
            }

            // -Y neighbor
            if (mask(0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i - 1, j);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = mask(0, 1);
                    c++;
                }
            }

            // +X neighbor
            if (mask(1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j + 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = mask(1, 2);
                    c++;
                }
            }

            // -X neighbor
            if (mask(1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j - 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = mask(1, 0);
                    c++;
                }
            }
        }
    }

    // Error checking
    if (c != v_nnz) {
        std::cerr << "ERROR: V component NNZ mismatch! Expected " << v_nnz << " but got " << c << std::endl;
    }
    
    for (int i = 0; i < v_nnz; ++i) {
        if (v_rows[i] < 0 || v_rows[i] >= v_dof_count || 
            v_collums[i] < 0 || v_collums[i] >= v_dof_count) {
            std::cerr << "ERROR V: Out of bounds at entry " << i 
                      << " - row=" << v_rows[i] 
                      << " col=" << v_collums[i] 
                      << " MatSize=" << v_dof_count << std::endl;
        }
    }

    // Build sparse matrix
    std::vector<Eigen::Triplet<double>> triplets_v;
    triplets_v.reserve(v_nnz);
    for (int i = 0; i < v_nnz; ++i)
    {
        triplets_v.push_back(Triplet(v_rows[i], v_collums[i], v_values[i]));
    }

    SparseMatrix mat_v(v_dof_count, v_dof_count);
    mat_v.setFromTriplets(triplets_v.begin(), triplets_v.end());
    mat_v = ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_v;

    SparseMatrix identity_v(v_dof_count, v_dof_count);
    identity_v.setIdentity();
    DiffusionSolver2D::V_DIFFUSION_MATRIX_EIGEN = identity_v - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_v;

    free(v_collums);
    free(v_rows);
    free(v_values);
}
void DiffusionSolver2D::SolveDiffusion_V_Eigen(MAC* gridAnt, MAC* gridSol){
    DiffusionSolver2D::Update_V_DiffusionMatrix(gridAnt);
    int v_matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();

    double dh = gridAnt->dh;

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-9);


     // ========== Solve V component ==========
    
    // Build RHS from previous timestep V velocities
    VectorXd RHS_V = VectorXd(v_matrix_size);
    RHS_V.setZero();
    
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1) continue;
            if (id >= v_matrix_size) {
                std::cerr << "ERROR: V RHS index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            RHS_V(id) = gridAnt->GetV(i, j);  // Previous timestep value
        }
    }
    
    solver.compute(V_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }
    
    Eigen::VectorXd SOL_V = solver.solve(RHS_V);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }
    
    std::cout << "V solve completed in " << solver.iterations() << " iterations" << std::endl;
    std::cout << "V solve error: " << solver.error() << std::endl;

    // Copy V solution back to grid
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1) continue;
            if (id >= v_matrix_size) {
                std::cerr << "ERROR: V solution index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            gridSol->SetV(i, j, SOL_V(id));
        }
    }
    


}

void DiffusionSolver2D::Update_U_DiffusionMatrix(MAC* grid){
    double dh = SIMULATION.dh;

    DiffusionSolver2D::U_IDP = VectorXd((Nx+1) * Ny);
    DiffusionSolver2D::U_IDP.setConstant(-1);  

    U_MASK.clear();

    
    // ========== U COMPONENT ==========
    
    // First pass: count DOFs and build masks
    int u_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx+1; j++)
        {
            Tensor<double, 2> mask = Tensor<double, 2>(3, 3);
            mask.setZero();
            
            // Boundary nodes
            if (i == 0 || j == 0 || j==1 || i == Ny-1 || j == Nx || j==Nx-1)
            {
                DiffusionSolver2D::U_MASK.push_back(mask);
                Set_U_IDP(i, j, -1);
                continue;
            }
            
            // Non-fluid face check
            if (grid->GetSolid(i, j) != FLUID_CELL && grid->GetSolid(i, j-1) != FLUID_CELL){
                DiffusionSolver2D::U_MASK.push_back(mask);
                Set_U_IDP(i, j, -1);
                continue;
            }


            // This is a valid DOF - build stencil
            if(grid->GetSolid(i+1, j) == FLUID_CELL || grid->GetSolid(i+1, j-1) == FLUID_CELL){
                mask(2, 1) = 1.0;
                mask(1, 1) -= 1.0;
            }
            else if(grid->GetSolid(i+1, j) == SOLID_CELL || grid->GetSolid(i+1, j-1) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            if(grid->GetSolid(i-1, j) == FLUID_CELL || grid->GetSolid(i-1, j-1) == FLUID_CELL){
                mask(0, 1) = 1.0;
                mask(1, 1) -= 1.0;
            }
            else if(grid->GetSolid(i-1, j) == SOLID_CELL || grid->GetSolid(i-1, j-1) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }
            

            if(grid->GetSolid(i, j) == FLUID_CELL){
                mask(1, 2) = 1.0;
                mask(1, 1) -= 1.0;
            }
            
            if(grid->GetSolid(i, j-1) == FLUID_CELL){
                mask(1, 0) = 1.0;
                mask(1, 1) -= 1.0;
            }


            
            if(grid->GetSolid(i, j) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }
            
            if(grid->GetSolid(i, j-1) == SOLID_CELL){
                mask(1, 1) -= 1.0;
            }

            Set_U_IDP(i, j, u_dof_count);
            u_dof_count++;
            DiffusionSolver2D::U_MASK.push_back(mask);
        }
    }

    // Second pass: count non-zeros
    int u_nnz = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            int MAT_LINE = Get_U_IDP(i, j);
            if (MAT_LINE == -1) continue;
            
            Tensor<double, 2> mask = GetUMask(i, j);
            
            if (mask(1, 1) != 0.0) u_nnz++;
            if (mask(2, 1) != 0.0 && Get_U_IDP(i + 1, j) != -1) u_nnz++;
            if (mask(0, 1) != 0.0 && Get_U_IDP(i - 1, j) != -1) u_nnz++;
            if (mask(1, 2) != 0.0 && Get_U_IDP(i, j + 1) != -1) u_nnz++;
            if (mask(1, 0) != 0.0 && Get_U_IDP(i, j - 1) != -1) u_nnz++;
        }
    }

    std::cout << "U component: DOFs = " << u_dof_count << ", NNZ = " << u_nnz << std::endl;

    // Allocate arrays
    u_collums = (int *)malloc(sizeof(int) * u_nnz);
    u_rows = (int *)malloc(sizeof(int) * u_nnz);
    u_values = (double *)calloc(u_nnz, sizeof(double));
    
    // Third pass: assemble matrix
    int c = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            Tensor<double, 2> mask = GetUMask(i, j);
            int MAT_LINE = Get_U_IDP(i, j);
            
            if (MAT_LINE == -1) continue;

            // Diagonal
            if (mask(1, 1) != 0.0)
            {
                u_rows[c] = MAT_LINE;
                u_collums[c] = MAT_LINE;
                u_values[c] = mask(1, 1);
                c++;
            }

            // +Y neighbor
            if (mask(2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i + 1, j);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = mask(2, 1);
                    c++;
                }
            }

            // -Y neighbor
            if (mask(0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i - 1, j);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = mask(0, 1);
                    c++;
                }
            }

            // +X neighbor
            if (mask(1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j + 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = mask(1, 2);
                    c++;
                }
            }

            // -X neighbor
            if (mask(1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j - 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = mask(1, 0);
                    c++;
                }
            }
        }
    }

    // Error checking
    if (c != u_nnz) {
        std::cerr << "ERROR: U component NNZ mismatch! Expected " << u_nnz << " but got " << c << std::endl;
    }
    
    for (int i = 0; i < u_nnz; ++i) {
        if (u_rows[i] < 0 || u_rows[i] >= u_dof_count || 
            u_collums[i] < 0 || u_collums[i] >= u_dof_count) {
            std::cerr << "ERROR U: Out of bounds at entry " << i 
                      << " - row=" << u_rows[i] 
                      << " col=" << u_collums[i] 
                      << " MatSize=" << u_dof_count << std::endl;
        }
    }

    // Build sparse matrix
    std::vector<Eigen::Triplet<double>> triplets_u;
    triplets_u.reserve(u_nnz);
    for (int i = 0; i < u_nnz; ++i)
    {
        triplets_u.push_back(Triplet(u_rows[i], u_collums[i], u_values[i]));
    }

    SparseMatrix mat_u(u_dof_count, u_dof_count);
    mat_u.setFromTriplets(triplets_u.begin(), triplets_u.end());
    SparseMatrix identity_u(u_dof_count, u_dof_count);
    identity_u.setIdentity();
    
    // Build implicit diffusion matrix: (I - dt*ε/dh² * L)
    DiffusionSolver2D::U_DIFFUSION_MATRIX_EIGEN = identity_u - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_u;

    free(u_collums);
    free(u_rows);
    free(u_values);

}
void DiffusionSolver2D::SolveDiffusion_U_Eigen(MAC* gridAnt, MAC* gridSol){
    Update_U_DiffusionMatrix(gridAnt);
    DiffusionSolver2D::dt = SIMULATION.dt;  
    int u_matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();
    double dh = gridAnt->dh;
    
    // Build RHS from previous timestep U velocities
    VectorXd RHS_U = VectorXd(u_matrix_size);
    RHS_U.setZero();
    
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1) continue;
            if (id >= u_matrix_size) {
                std::cerr << "ERROR: U RHS index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            RHS_U(id) = gridAnt->GetU(i, j);  // Previous timestep value
        }
    }

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-9);
    solver.compute(U_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: U matrix decomposition failed!" << std::endl;
        return;
    }
    
    Eigen::VectorXd SOL_U = solver.solve(RHS_U);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: U solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }
    
    std::cout << "U solve completed in " << solver.iterations() << " iterations" << std::endl;
    std::cout << "U solve error: " << solver.error() << std::endl;

    // Copy U solution back to grid
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1) continue;
            if (id >= u_matrix_size) {
                std::cerr << "ERROR: U solution index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            gridSol->SetU(i, j, SOL_U(id));
        }
    }

}




void DiffusionSolver2D::SolveDiffusion_Eigen(MAC* gridAnt, MAC* gridSol){

    DiffusionSolver2D::dt = SIMULATION.dt;
    Eigen::setNbThreads(THREAD_COUNT);
    
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            SolveDiffusion_U_Eigen(gridAnt, gridSol);

            #pragma omp task
            SolveDiffusion_V_Eigen(gridAnt, gridSol);
        }
    }

    double end = omp_get_wtime();
    SIMULATION.lastADISolveTime = end - start;

    /*
    DiffusionSolver2D::UpdatePressureMatrix(gridAnt);
    std::cout << "Solving..." << std::endl;
    CPUTimer timer;
    timer.start();
    double dh = gridAnt->dh;
    
    // Get the actual matrix sizes
    int u_matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();
    int v_matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();
    
    std::cout << "U matrix size: " << u_matrix_size << std::endl;
    std::cout << "V matrix size: " << v_matrix_size << std::endl;

    // ========== Solve U component ==========
    
    // Build RHS from previous timestep U velocities
    VectorXd RHS_U = VectorXd(u_matrix_size);
    RHS_U.setZero();
    
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1) continue;
            if (id >= u_matrix_size) {
                std::cerr << "ERROR: U RHS index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            RHS_U(id) = gridAnt->GetU(i, j);  // Previous timestep value
        }
    }

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-12);
    solver.compute(U_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: U matrix decomposition failed!" << std::endl;
        return;
    }
    
    Eigen::VectorXd SOL_U = solver.solve(RHS_U);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: U solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }
    
    std::cout << "U solve completed in " << solver.iterations() << " iterations" << std::endl;
    std::cout << "U solve error: " << solver.error() << std::endl;

    // Copy U solution back to grid
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1) continue;
            if (id >= u_matrix_size) {
                std::cerr << "ERROR: U solution index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            gridSol->SetU(i, j, SOL_U(id));
        }
    }

    // ========== Solve V component ==========
    
    // Build RHS from previous timestep V velocities
    VectorXd RHS_V = VectorXd(v_matrix_size);
    RHS_V.setZero();
    
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1) continue;
            if (id >= v_matrix_size) {
                std::cerr << "ERROR: V RHS index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            RHS_V(id) = gridAnt->GetV(i, j);  // Previous timestep value
        }
    }
    
    solver.compute(V_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }
    
    Eigen::VectorXd SOL_V = solver.solve(RHS_V);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }
    
    std::cout << "V solve completed in " << solver.iterations() << " iterations" << std::endl;
    std::cout << "V solve error: " << solver.error() << std::endl;

    // Copy V solution back to grid
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1) continue;
            if (id >= v_matrix_size) {
                std::cerr << "ERROR: V solution index out of bounds! id=" << id 
                          << " at (" << i << "," << j << ")" << std::endl;
                continue;
            }
            gridSol->SetV(i, j, SOL_V(id));
        }
    }
    
    timer.stop();
    std::cout << "Diffusion solve completed!" << std::endl;
    */
}
