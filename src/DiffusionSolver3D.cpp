#include "headers/Solvers/DiffusionSolver3D.h"


// Static member initialization for DiffusionSolver3D

// Grid dimensions
int DiffusionSolver3D::Nx = 0;
int DiffusionSolver3D::Ny = 0;
int DiffusionSolver3D::Nz = 0;

// Time step
double DiffusionSolver3D::dt = 0.0;

// COO format arrays for U
int* DiffusionSolver3D::u_collums = nullptr;
int* DiffusionSolver3D::u_rows = nullptr;
double* DiffusionSolver3D::u_values = nullptr;

// COO format arrays for V
int* DiffusionSolver3D::v_collums = nullptr;
int* DiffusionSolver3D::v_rows = nullptr;
double* DiffusionSolver3D::v_values = nullptr;

// COO format arrays for W
int* DiffusionSolver3D::w_collums = nullptr;
int* DiffusionSolver3D::w_rows = nullptr;
double* DiffusionSolver3D::w_values = nullptr;

// Sparse matrices
SparseMatrix DiffusionSolver3D::U_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver3D::V_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver3D::W_DIFFUSION_MATRIX_EIGEN;

// Index mapping vectors
VectorXd DiffusionSolver3D::U_IDP;
VectorXd DiffusionSolver3D::V_IDP;
VectorXd DiffusionSolver3D::W_IDP;

// Mask vectors (3x3x3 stencils for 3D)
std::vector<Tensor<double,3>> DiffusionSolver3D::U_MASK;
std::vector<Tensor<double,3>> DiffusionSolver3D::V_MASK;
std::vector<Tensor<double,3>> DiffusionSolver3D::W_MASK;

void DiffusionSolver3D::InitializeDiffusionSolver(MAC* grid){
    DiffusionSolver3D::Nx = SIMULATION.Nx;
    DiffusionSolver3D::Ny = SIMULATION.Ny;
    DiffusionSolver3D::Nz = SIMULATION.Nz;
    DiffusionSolver3D::U_MASK.reserve(grid->GetFluidCellCount()+1);
    DiffusionSolver3D::V_MASK.reserve(grid->GetFluidCellCount()+1);
    DiffusionSolver3D::W_MASK.reserve(grid->GetFluidCellCount()+1);
    DiffusionSolver3D::dt = SIMULATION.dt;
    
    DiffusionSolver3D::U_IDP = VectorXd((Nx+1) * Ny * Nz);
    DiffusionSolver3D::U_IDP.setConstant(-1);  
    DiffusionSolver3D::V_IDP = VectorXd(Nx * (Ny+1) * Nz);
    DiffusionSolver3D::V_IDP.setConstant(-1);  
    DiffusionSolver3D::W_IDP = VectorXd(Nx * Ny * (Nz+1));
    DiffusionSolver3D::W_IDP.setConstant(-1);  

    //UpdatePressureMatrix(grid);
}

void DiffusionSolver3D::UpdatePressureMatrix(MAC* grid){
    //solving V first for debug
    double dh = SIMULATION.dh;
    DiffusionSolver3D::dt = SIMULATION.dt;
    DiffusionSolver3D::U_IDP = VectorXd((Nx+1) * (Ny) *(Nz));
    DiffusionSolver3D::U_IDP.setConstant(-1);  
    DiffusionSolver3D::V_IDP = VectorXd((Nx) * (Ny+1) *(Nz));
    DiffusionSolver3D::V_IDP.setConstant(-1);  
    DiffusionSolver3D::W_IDP = VectorXd((Nx) * (Ny) *(Nz+1));
    DiffusionSolver3D::W_IDP.setConstant(-1);  
    U_MASK.clear();
    V_MASK.clear();
    W_MASK.clear();

    int c = 0;
    
    

    //setting the U mask up
    int u_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx+1; j++)
        {
            for(int k = 0;k< Nz;k++){
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3,3);
                mask.setZero();
                if (i == 0 || j == 0 || j==1 || i == Ny || j == Nx || j==Nx-1 || k == 0 || k == Nz)
                {
                    DiffusionSolver3D::U_MASK.push_back(mask);
                    Set_U_IDP(i, j,k, -1);
                    continue;
                }

                // Non-fluid face check
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i, j-1,k) != FLUID_CELL){
                    DiffusionSolver3D::U_MASK.push_back(mask);
                    Set_U_IDP(i, j,k, -1);
                    continue;
                }

                //if we get here, we have a valid dof, so we can built a stencil
                //is any of our top neighboors a fluid cell? 
                //if yes, we should solve him
                if(grid->GetSolid(i+1, j,k) == FLUID_CELL || grid->GetSolid(i+1, j-1,k) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //if not, is he a solid cell?, if yes, he is a dirichlet node (0)
                else if(grid->GetSolid(i+1, j,k) == SOLID_CELL || grid->GetSolid(i+1, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now the same for our botton one
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL || grid->GetSolid(i-1, j-1,k) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i-1, j,k) == SOLID_CELL || grid->GetSolid(i-1, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now we apply the exact same logic for our front neighboor
                if(grid->GetSolid(i, j,k+1) == FLUID_CELL || grid->GetSolid(i, j-1,k+1) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j,k+1) == SOLID_CELL || grid->GetSolid(i, j-1,k+1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //and our back
                if(grid->GetSolid(i, j,k-1) == FLUID_CELL || grid->GetSolid(i, j-1,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j,k-1) == SOLID_CELL || grid->GetSolid(i, j-1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now our right and left is simpler since we sit on the face of those cells, so we check for fluid
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j-1,k) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //and solid
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                Set_U_IDP(i, j,k, u_dof_count);
                u_dof_count++;
                DiffusionSolver3D::U_MASK.push_back(mask);


            }

        }
    }

    //now we count our non zero matrix entries... we can actually not do this but my mental faculties are declining as
    //i write this code
    int u_nnz = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int MAT_LINE = Get_U_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                Tensor<double, 3> mask = GetUMask(i, j,k);
            
                if (mask(1, 1,1) != 0.0) u_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_U_IDP(i + 1, j,k) != -1) u_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_U_IDP(i - 1, j,k) != -1) u_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_U_IDP(i, j + 1,k) != -1) u_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_U_IDP(i, j - 1,k) != -1) u_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_U_IDP(i, j,k + 1) != -1) u_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_U_IDP(i, j,k - 1) != -1) u_nnz++;

            }

        }
    }
    std::cout << "U component: DOFs = " << u_dof_count << ", NNZ = " << u_nnz << std::endl;
    //finally, we assemble the matrix
    u_collums = (int *)malloc(sizeof(int) * u_nnz);
    u_rows = (int *)malloc(sizeof(int) * u_nnz);
    u_values = (double *)calloc(u_nnz, sizeof(double));
    c = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                Tensor<double, 3> mask = GetUMask(i, j,k);
                int MAT_LINE = Get_U_IDP(i, j,k);
                if (MAT_LINE == -1) continue;
                //the diag
                if (mask(1, 1,1) != 0.0)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_LINE;
                    u_values[c] = mask(1, 1,1);
                    c++;
                }

                //the top (+Y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(0, 1,1);
                        c++;
                    }
                }
                //right (+x)
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 2,1);
                        c++;
                    }
                }
                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 0,1);
                        c++;
                    }
                }
                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 1,2);
                        c++;
                    }
                }
                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 1,0);
                        c++;
                    }
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
    
    // implicfit difculsili mat
    DiffusionSolver3D::U_DIFFUSION_MATRIX_EIGEN = identity_u - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_u;

    free(u_collums);
    free(u_rows);
    free(u_values);  
    

    

    
    //setting the V mask up
    int v_dof_count = 0;
    c = 0;

    for (int i = 0; i < Ny+1; i++){
        for (int j = 0; j < Nx; j++){
            for(int k = 0;k<Nz;k++ ){

                Tensor<double,3> mask = Tensor<double, 3>(3,3,3);
                mask.setZero();
                //if it is a boundary node, we do not solve
                if(i == 0 || j == 0 || i == 1 || i == Ny || i == Ny-1 ||j == Nx || k == 0 || k ==  Nz){
                    DiffusionSolver3D::V_MASK.push_back(mask);
                    Set_V_IDP(i,j,k,-1);
                    continue;
                }

                //if it is a non fluid face, we do not solve (either all solid or completely empty)
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i-1, j,k) != FLUID_CELL){
                    DiffusionSolver3D::V_MASK.push_back(mask);
                    Set_V_IDP(i, j,k, -1);
                    continue;
                 }

                //to our right
                if(grid->GetSolid(i-1, j+1,k) == FLUID_CELL || grid->GetSolid(i, j+1,k) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                //check our left
                else if(grid->GetSolid(i-1, j-1,k) == SOLID_CELL || grid->GetSolid(i, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //to our left
                if(grid->GetSolid(i-1, j-1,k) == FLUID_CELL || grid->GetSolid(i, j-1,k) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                //ceck our right
                else if(grid->GetSolid(i-1, j+1,k) == SOLID_CELL || grid->GetSolid(i, j+1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                //our front
                if(grid->GetSolid(i-1, j,k+1) == FLUID_CELL || grid->GetSolid(i, j,k+1) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //check our back
                else if(grid->GetSolid(i-1, j,k-1) == SOLID_CELL || grid->GetSolid(i, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our back
                if(grid->GetSolid(i-1, j,k-1) == FLUID_CELL || grid->GetSolid(i, j,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //check our front
                else if(grid->GetSolid(i-1, j,k+1) == SOLID_CELL || grid->GetSolid(i, j,k+1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our top block
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //Botton block
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //top
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our botton one
                if(grid->GetSolid(i-1, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

            
                Set_V_IDP(i, j, k,v_dof_count);
                v_dof_count++;
                DiffusionSolver3D::V_MASK.push_back(mask);

            }

        }
    }

    //now count the zeros
     int v_nnz = 0;
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k =1;k<Nz-1;k++){

                int MAT_LINE = Get_V_IDP(i, j,k);
                if (MAT_LINE == -1) continue;
                
                Tensor<double, 3> mask = GetVMask(i, j,k);

                if (mask(1, 1,1) != 0.0) v_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_V_IDP(i + 1, j,k) != -1) v_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_V_IDP(i - 1, j,k) != -1) v_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_V_IDP(i, j + 1,k) != -1) v_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_V_IDP(i, j - 1,k) != -1) v_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_V_IDP(i, j ,k+1) != -1) v_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_V_IDP(i, j,k-1) != -1) v_nnz++;
            }

        }
    }
    std::cout << "V component: DOFs = " << v_dof_count << ", NNZ = " << v_nnz << std::endl;
    //finally, assemble the matrix
    v_collums = (int *)malloc(sizeof(int) * v_nnz);
    v_rows = (int *)malloc(sizeof(int) * v_nnz);
    v_values = (double *)calloc(v_nnz, sizeof(double));

    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++)
            {
                Tensor<double, 3> mask = GetVMask(i, j,k);
                int MAT_LINE = Get_V_IDP(i, j,k);
                //this is not a valid line, sicne we dont solve it
                if (MAT_LINE == -1) continue;
                //the diagonal
                if (mask(1, 1,1) != 0.0)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_LINE;
                    v_values[c] = mask(1, 1,1);
                    c++;
                }

                //top (+y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(0, 1,1);
                        c++;
                    }
                }

                //right (+x)
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 2,1);
                        c++;
                    }
                }
                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 0,1);
                        c++;
                    }
                }

                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 1,2);
                        c++;
                    }
                }

                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 1,0);
                        c++;
                    }
                }


            }

        }

    }

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
    DiffusionSolver3D::V_DIFFUSION_MATRIX_EIGEN = identity_v - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_v;

    free(v_collums);
    free(v_rows);
    free(v_values);

    //finally, we now make the w mask

    
    int w_dof_count = 0;
    c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for(int k = 0;k< Nz+1;k++){
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3,3);
                mask.setZero();
                if (i == 0 || j == 0  || i == Ny || j == Nx  || k == 0 || k == Nz || k == 1 || k == Nz-1){
                    DiffusionSolver3D::W_MASK.push_back(mask);
                    Set_W_IDP(i, j,k, -1);
                    continue;
                }

                //check if this face needs solving
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i, j,k-1) != FLUID_CELL){
                    DiffusionSolver3D::W_MASK.push_back(mask);
                    Set_W_IDP(i, j,k, -1);
                    continue;
                }
                //Same dance here, if we get here, we have a valid dof
                //we now ceheck for our top neighboors
                if(grid->GetSolid(i+1, j,k) == FLUID_CELL || grid->GetSolid(i+1, j,k-1) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //if not, is he a solid cell?, if yes, he is a dirichlet node (0)
                else if(grid->GetSolid(i+1, j,k) == SOLID_CELL || grid->GetSolid(i+1, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                //now botton
                //now the same for our botton one
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL || grid->GetSolid(i-1, j,k-1) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i-1, j,k) == SOLID_CELL || grid->GetSolid(i-1, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }


                //now right
                if(grid->GetSolid(i, j+1,k) == FLUID_CELL || grid->GetSolid(i, j+1,k-1) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                else if(grid->GetSolid(i, j+1,k) == SOLID_CELL || grid->GetSolid(i, j+1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //and left
                if(grid->GetSolid(i, j-1,k) == FLUID_CELL || grid->GetSolid(i, j-1,k-1) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j-1,k) == SOLID_CELL || grid->GetSolid(i, j-1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //finally, fron and back is simpler since we sit on the face of a cell
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //and solid
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                Set_W_IDP(i, j,k, w_dof_count);
                w_dof_count++;
                DiffusionSolver3D::W_MASK.push_back(mask);

            }

        }
    }
    //counting
    int w_nnz = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int MAT_LINE = Get_W_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                Tensor<double, 3> mask = GetWMask(i, j,k);
                if (mask(1, 1,1) != 0.0) w_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_W_IDP(i + 1, j,k) != -1) w_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_W_IDP(i - 1, j,k) != -1) w_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_W_IDP(i, j + 1,k) != -1) w_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_W_IDP(i, j - 1,k) != -1) w_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_W_IDP(i, j,k + 1) != -1) w_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_W_IDP(i, j,k - 1) != -1) w_nnz++;

            }

        }
    }

    //finally, matrix assembly

    std::cout << "W component: DOFs = " << w_dof_count << ", NNZ = " << w_nnz << std::endl;

    w_collums = (int *)malloc(sizeof(int) * w_nnz);
    w_rows = (int *)malloc(sizeof(int) * w_nnz);
    w_values = (double *)calloc(w_nnz, sizeof(double)); 
    c = 0;    
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                Tensor<double, 3> mask = GetWMask(i, j,k);
                int MAT_LINE = Get_W_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                if (mask(1, 1,1) != 0.0)
                {
                    w_rows[c] = MAT_LINE;
                    w_collums[c] = MAT_LINE;
                    w_values[c] = mask(1, 1,1);
                    c++;
                }
                //the top (+Y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(0, 1,1);
                        c++;
                    }
                }
                //right
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 2,1);
                        c++;
                    }
                }

                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 0,1);
                        c++;
                    }
                }
                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 1,2);
                        c++;
                    }
                }
                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 1,0);
                        c++;
                    }
                }



            }
        }
    }

    if (c != w_nnz) {
        std::cerr << "ERROR: W component NNZ mismatch! Expected " << w_nnz << " but got " << c << std::endl;
    }

    for (int i = 0; i < w_nnz; ++i) {
        if (w_rows[i] < 0 || w_rows[i] >= w_dof_count || 
            w_collums[i] < 0 || w_collums[i] >= w_dof_count) {
            std::cerr << "ERROR W: Out of bounds at entry " << i 
                      << " - row=" << w_rows[i] 
                      << " col=" << w_collums[i] 
                      << " MatSize=" << w_dof_count << std::endl;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets_w;
    triplets_w.reserve(w_nnz);
    for (int i = 0; i < w_nnz; ++i)
    {
        triplets_w.push_back(Triplet(w_rows[i], w_collums[i], w_values[i]));
    }

    SparseMatrix mat_w(w_dof_count, w_dof_count);
    mat_w.setFromTriplets(triplets_w.begin(), triplets_w.end());
    SparseMatrix identity_w(w_dof_count, w_dof_count);
    identity_w.setIdentity();
    
    // implicfit difculsili mat
    DiffusionSolver3D::W_DIFFUSION_MATRIX_EIGEN = identity_w - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_w;

    free(w_collums);
    free(w_rows);
    free(w_values);  
    


}

//this is highly parelalizable, we can assemble and solve each matrix completely separated
void DiffusionSolver3D::SolveDiffusion_Eigen(MAC* gridAnt, MAC* gridSol){
    DiffusionSolver3D::dt = SIMULATION.dt;
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
            #pragma omp task
            SolveDiffusion_W_Eigen(gridAnt, gridSol);
        }
    }

 
    double end = omp_get_wtime();
    SIMULATION.lastADISolveTime = end - start;


    /*
    DiffusionSolver3D::UpdatePressureMatrix(gridAnt);
    std::cout << "Solving..." << std::endl;
    DiffusionSolver3D::dt = SIMULATION.dt;  
    double dh = gridAnt->dh;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-12);
    // ========== Solve U component ==========
    int u_matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();
    std::cout << "U matrix size: " << u_matrix_size << std::endl;
    VectorXd RHS_U = VectorXd(u_matrix_size);
    RHS_U.setZero();
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k  = 1;k<Nz;k++){
                int id = Get_U_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= u_matrix_size) {
                    std::cerr << "ERROR: U RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_U(id) = gridAnt->GetU(i, j,k);
            }

        }
    }

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
    //copy it back
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k  = 1;k<Nz;k++){
                int id = Get_U_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= u_matrix_size) {
                    std::cerr << "ERROR: U solution index out of bounds! id=" << id   << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetU(i, j, k,SOL_U(id));

            }
        }
    }



    
    // ========== Solve V component ==========
    int v_matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();
    std::cout << "V matrix size: " << v_matrix_size << std::endl;
    VectorXd RHS_V = VectorXd(v_matrix_size);
    RHS_V.setZero();
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int id = Get_V_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= v_matrix_size) {
                    std::cerr << "ERROR: V RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_V(id) = gridAnt->GetV(i, j,k);  // Previous timestep value
            }
        }
    }

    solver.compute(V_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }
    Eigen::VectorXd SOL_V = solver.solve(RHS_V);


    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int id = Get_V_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= v_matrix_size) {
                    std::cerr << "ERROR: V solution index out of bounds! id=" << id     << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetV(i, j,k, SOL_V(id));
            }
        }
        
    }


    //============= W Solve

    int w_matrix_size = W_DIFFUSION_MATRIX_EIGEN.rows();
    std::cout << "W matrix size: " << w_matrix_size << std::endl;
    VectorXd RHS_W = VectorXd(w_matrix_size);
    RHS_W.setZero();
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int id = Get_W_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= w_matrix_size) {
                    std::cerr << "ERROR: W RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_W(id) = gridAnt->GetW(i, j,k);  // Previous timestep value
            }
        }
    }

    solver.compute(W_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: W matrix decomposition failed!" << std::endl;
        return;
    }
    Eigen::VectorXd SOL_W = solver.solve(RHS_W);


    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int id = Get_W_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= w_matrix_size) {
                    std::cerr << "ERROR: W solution index out of bounds! id=" << id     << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetW(i, j,k, SOL_W(id));
            }
        }
    }
        */
    

    
}




void DiffusionSolver3D::Update_U_DiffusionMatrix(MAC* grid){
    double dh = SIMULATION.dh;
    DiffusionSolver3D::U_IDP = VectorXd((Nx+1) * (Ny) *(Nz));
    DiffusionSolver3D::U_IDP.setConstant(-1);  
    U_MASK.clear();


    int c = 0;
    
    

    //setting the U mask up
    int u_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx+1; j++)
        {
            for(int k = 0;k< Nz;k++){
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3,3);
                mask.setZero();
                if (i == 0 || j == 0 || j==1 || i == Ny || j == Nx || j==Nx-1 || k == 0 || k == Nz)
                {
                    DiffusionSolver3D::U_MASK.push_back(mask);
                    Set_U_IDP(i, j,k, -1);
                    continue;
                }

                // Non-fluid face check
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i, j-1,k) != FLUID_CELL){
                    DiffusionSolver3D::U_MASK.push_back(mask);
                    Set_U_IDP(i, j,k, -1);
                    continue;
                }

                //if we get here, we have a valid dof, so we can built a stencil
                //is any of our top neighboors a fluid cell? 
                //if yes, we should solve him
                if(grid->GetSolid(i+1, j,k) == FLUID_CELL || grid->GetSolid(i+1, j-1,k) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //if not, is he a solid cell?, if yes, he is a dirichlet node (0)
                else if(grid->GetSolid(i+1, j,k) == SOLID_CELL || grid->GetSolid(i+1, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now the same for our botton one
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL || grid->GetSolid(i-1, j-1,k) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i-1, j,k) == SOLID_CELL || grid->GetSolid(i-1, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now we apply the exact same logic for our front neighboor
                if(grid->GetSolid(i, j,k+1) == FLUID_CELL || grid->GetSolid(i, j-1,k+1) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j,k+1) == SOLID_CELL || grid->GetSolid(i, j-1,k+1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //and our back
                if(grid->GetSolid(i, j,k-1) == FLUID_CELL || grid->GetSolid(i, j-1,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j,k-1) == SOLID_CELL || grid->GetSolid(i, j-1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //now our right and left is simpler since we sit on the face of those cells, so we check for fluid
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j-1,k) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //and solid
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                Set_U_IDP(i, j,k, u_dof_count);
                u_dof_count++;
                DiffusionSolver3D::U_MASK.push_back(mask);


            }

        }
    }

    //now we count our non zero matrix entries... we can actually not do this but my mental faculties are declining as
    //i write this code
    int u_nnz = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int MAT_LINE = Get_U_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                Tensor<double, 3> mask = GetUMask(i, j,k);
            
                if (mask(1, 1,1) != 0.0) u_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_U_IDP(i + 1, j,k) != -1) u_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_U_IDP(i - 1, j,k) != -1) u_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_U_IDP(i, j + 1,k) != -1) u_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_U_IDP(i, j - 1,k) != -1) u_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_U_IDP(i, j,k + 1) != -1) u_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_U_IDP(i, j,k - 1) != -1) u_nnz++;

            }

        }
    }
    std::cout << "U component: DOFs = " << u_dof_count << ", NNZ = " << u_nnz << std::endl;
    //finally, we assemble the matrix
    u_collums = (int *)malloc(sizeof(int) * u_nnz);
    u_rows = (int *)malloc(sizeof(int) * u_nnz);
    u_values = (double *)calloc(u_nnz, sizeof(double));
    c = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                Tensor<double, 3> mask = GetUMask(i, j,k);
                int MAT_LINE = Get_U_IDP(i, j,k);
                if (MAT_LINE == -1) continue;
                //the diag
                if (mask(1, 1,1) != 0.0)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_LINE;
                    u_values[c] = mask(1, 1,1);
                    c++;
                }

                //the top (+Y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(0, 1,1);
                        c++;
                    }
                }
                //right (+x)
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 2,1);
                        c++;
                    }
                }
                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 0,1);
                        c++;
                    }
                }
                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 1,2);
                        c++;
                    }
                }
                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_U_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        u_rows[c] = MAT_LINE;
                        u_collums[c] = MAT_COLLUM;
                        u_values[c] = mask(1, 1,0);
                        c++;
                    }
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
    
    // implicfit difculsili mat
    DiffusionSolver3D::U_DIFFUSION_MATRIX_EIGEN = identity_u - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_u;

    free(u_collums);
    free(u_rows);
    free(u_values);  
    

    

};           
void DiffusionSolver3D::Update_V_DiffusionMatrix(MAC* grid){
    double dh = SIMULATION.dh;


    DiffusionSolver3D::V_IDP = VectorXd((Nx) * (Ny+1) *(Nz));
    DiffusionSolver3D::V_IDP.setConstant(-1);  


    V_MASK.clear();


    int c = 0;
    int v_dof_count = 0;


    for (int i = 0; i < Ny+1; i++){
        for (int j = 0; j < Nx; j++){
            for(int k = 0;k<Nz;k++ ){

                Tensor<double,3> mask = Tensor<double, 3>(3,3,3);
                mask.setZero();
                //if it is a boundary node, we do not solve
                if(i == 0 || j == 0 || i == 1 || i == Ny || i == Ny-1 ||j == Nx || k == 0 || k ==  Nz){
                    DiffusionSolver3D::V_MASK.push_back(mask);
                    Set_V_IDP(i,j,k,-1);
                    continue;
                }

                //if it is a non fluid face, we do not solve (either all solid or completely empty)
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i-1, j,k) != FLUID_CELL){
                    DiffusionSolver3D::V_MASK.push_back(mask);
                    Set_V_IDP(i, j,k, -1);
                    continue;
                 }

                //to our right
                if(grid->GetSolid(i-1, j+1,k) == FLUID_CELL || grid->GetSolid(i, j+1,k) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                //check our left
                else if(grid->GetSolid(i-1, j-1,k) == SOLID_CELL || grid->GetSolid(i, j-1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //to our left
                if(grid->GetSolid(i-1, j-1,k) == FLUID_CELL || grid->GetSolid(i, j-1,k) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                //ceck our right
                else if(grid->GetSolid(i-1, j+1,k) == SOLID_CELL || grid->GetSolid(i, j+1,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                //our front
                if(grid->GetSolid(i-1, j,k+1) == FLUID_CELL || grid->GetSolid(i, j,k+1) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //check our back
                else if(grid->GetSolid(i-1, j,k-1) == SOLID_CELL || grid->GetSolid(i, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our back
                if(grid->GetSolid(i-1, j,k-1) == FLUID_CELL || grid->GetSolid(i, j,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //check our front
                else if(grid->GetSolid(i-1, j,k+1) == SOLID_CELL || grid->GetSolid(i, j,k+1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our top block
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //Botton block
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //top
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //our botton one
                if(grid->GetSolid(i-1, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

            
                Set_V_IDP(i, j, k,v_dof_count);
                v_dof_count++;
                DiffusionSolver3D::V_MASK.push_back(mask);

            }

        }
    }

    //now count the zeros
     int v_nnz = 0;
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k =1;k<Nz-1;k++){

                int MAT_LINE = Get_V_IDP(i, j,k);
                if (MAT_LINE == -1) continue;
                
                Tensor<double, 3> mask = GetVMask(i, j,k);

                if (mask(1, 1,1) != 0.0) v_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_V_IDP(i + 1, j,k) != -1) v_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_V_IDP(i - 1, j,k) != -1) v_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_V_IDP(i, j + 1,k) != -1) v_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_V_IDP(i, j - 1,k) != -1) v_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_V_IDP(i, j ,k+1) != -1) v_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_V_IDP(i, j,k-1) != -1) v_nnz++;
            }

        }
    }
    std::cout << "V component: DOFs = " << v_dof_count << ", NNZ = " << v_nnz << std::endl;
    //finally, assemble the matrix
    v_collums = (int *)malloc(sizeof(int) * v_nnz);
    v_rows = (int *)malloc(sizeof(int) * v_nnz);
    v_values = (double *)calloc(v_nnz, sizeof(double));

    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++)
            {
                Tensor<double, 3> mask = GetVMask(i, j,k);
                int MAT_LINE = Get_V_IDP(i, j,k);
                //this is not a valid line, sicne we dont solve it
                if (MAT_LINE == -1) continue;
                //the diagonal
                if (mask(1, 1,1) != 0.0)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_LINE;
                    v_values[c] = mask(1, 1,1);
                    c++;
                }

                //top (+y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(0, 1,1);
                        c++;
                    }
                }

                //right (+x)
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 2,1);
                        c++;
                    }
                }
                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 0,1);
                        c++;
                    }
                }

                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 1,2);
                        c++;
                    }
                }

                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_V_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        v_rows[c] = MAT_LINE;
                        v_collums[c] = MAT_COLLUM;
                        v_values[c] = mask(1, 1,0);
                        c++;
                    }
                }


            }

        }

    }

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
    DiffusionSolver3D::V_DIFFUSION_MATRIX_EIGEN = identity_v - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_v;

    free(v_collums);
    free(v_rows);
    free(v_values);



};      
void DiffusionSolver3D::Update_W_DiffusionMatrix(MAC* grid){
    double dh = SIMULATION.dh;

    DiffusionSolver3D::W_IDP = VectorXd((Nx) * (Ny) *(Nz+1));
    DiffusionSolver3D::W_IDP.setConstant(-1);  

    W_MASK.clear();

    int c = 0;
    
    int w_dof_count = 0;
    c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for(int k = 0;k< Nz+1;k++){
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3,3);
                mask.setZero();
                if (i == 0 || j == 0  || i == Ny || j == Nx  || k == 0 || k == Nz || k == 1 || k == Nz-1){
                    DiffusionSolver3D::W_MASK.push_back(mask);
                    Set_W_IDP(i, j,k, -1);
                    continue;
                }

                //check if this face needs solving
                if (grid->GetSolid(i, j,k) != FLUID_CELL && grid->GetSolid(i, j,k-1) != FLUID_CELL){
                    DiffusionSolver3D::W_MASK.push_back(mask);
                    Set_W_IDP(i, j,k, -1);
                    continue;
                }
                //Same dance here, if we get here, we have a valid dof
                //we now ceheck for our top neighboors
                if(grid->GetSolid(i+1, j,k) == FLUID_CELL || grid->GetSolid(i+1, j,k-1) == FLUID_CELL){
                    mask(2, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //if not, is he a solid cell?, if yes, he is a dirichlet node (0)
                else if(grid->GetSolid(i+1, j,k) == SOLID_CELL || grid->GetSolid(i+1, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                //now botton
                //now the same for our botton one
                if(grid->GetSolid(i-1, j,k) == FLUID_CELL || grid->GetSolid(i-1, j,k-1) == FLUID_CELL){
                    mask(0, 1,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i-1, j,k) == SOLID_CELL || grid->GetSolid(i-1, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }


                //now right
                if(grid->GetSolid(i, j+1,k) == FLUID_CELL || grid->GetSolid(i, j+1,k-1) == FLUID_CELL){
                    mask(1, 2,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                else if(grid->GetSolid(i, j+1,k) == SOLID_CELL || grid->GetSolid(i, j+1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //and left
                if(grid->GetSolid(i, j-1,k) == FLUID_CELL || grid->GetSolid(i, j-1,k-1) == FLUID_CELL){
                    mask(1, 0,1) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                else if(grid->GetSolid(i, j-1,k) == SOLID_CELL || grid->GetSolid(i, j-1,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                //finally, fron and back is simpler since we sit on the face of a cell
                if(grid->GetSolid(i, j,k) == FLUID_CELL){
                    mask(1, 1,2) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j,k-1) == FLUID_CELL){
                    mask(1, 1,0) = 1.0;
                    mask(1, 1,1) -= 1.0;
                }
                //and solid
                if(grid->GetSolid(i, j,k) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }

                if(grid->GetSolid(i, j,k-1) == SOLID_CELL){
                    mask(1, 1,1) -= 1.0;
                }
                Set_W_IDP(i, j,k, w_dof_count);
                w_dof_count++;
                DiffusionSolver3D::W_MASK.push_back(mask);

            }

        }
    }
    //counting
    int w_nnz = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int MAT_LINE = Get_W_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                Tensor<double, 3> mask = GetWMask(i, j,k);
                if (mask(1, 1,1) != 0.0) w_nnz++;
                if (mask(2, 1,1) != 0.0 && Get_W_IDP(i + 1, j,k) != -1) w_nnz++;
                if (mask(0, 1,1) != 0.0 && Get_W_IDP(i - 1, j,k) != -1) w_nnz++;
                if (mask(1, 2,1) != 0.0 && Get_W_IDP(i, j + 1,k) != -1) w_nnz++;
                if (mask(1, 0,1) != 0.0 && Get_W_IDP(i, j - 1,k) != -1) w_nnz++;
                if (mask(1, 1,2) != 0.0 && Get_W_IDP(i, j,k + 1) != -1) w_nnz++;
                if (mask(1, 1,0) != 0.0 && Get_W_IDP(i, j,k - 1) != -1) w_nnz++;

            }

        }
    }

    //finally, matrix assembly

    std::cout << "W component: DOFs = " << w_dof_count << ", NNZ = " << w_nnz << std::endl;

    w_collums = (int *)malloc(sizeof(int) * w_nnz);
    w_rows = (int *)malloc(sizeof(int) * w_nnz);
    w_values = (double *)calloc(w_nnz, sizeof(double)); 
    c = 0;    
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                Tensor<double, 3> mask = GetWMask(i, j,k);
                int MAT_LINE = Get_W_IDP(i, j,k);
                if (MAT_LINE == -1) continue;

                if (mask(1, 1,1) != 0.0)
                {
                    w_rows[c] = MAT_LINE;
                    w_collums[c] = MAT_LINE;
                    w_values[c] = mask(1, 1,1);
                    c++;
                }
                //the top (+Y)
                if (mask(2, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i + 1, j,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(2, 1,1);
                        c++;
                    }
                }
                //botton (-y)
                if (mask(0, 1,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i -1 , j,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(0, 1,1);
                        c++;
                    }
                }
                //right
                if (mask(1, 2,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j+1,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 2,1);
                        c++;
                    }
                }

                //left (-x)
                if (mask(1, 0,1) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j-1,k);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 0,1);
                        c++;
                    }
                }
                //front (+z)
                if (mask(1, 1,2) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j,k+1);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 1,2);
                        c++;
                    }
                }
                //back (-z)
                if (mask(1, 1,0) != 0.0)
                {
                    int MAT_COLLUM = Get_W_IDP(i , j,k-1);
                    if (MAT_COLLUM != -1)
                    {
                        w_rows[c] = MAT_LINE;
                        w_collums[c] = MAT_COLLUM;
                        w_values[c] = mask(1, 1,0);
                        c++;
                    }
                }



            }
        }
    }

    if (c != w_nnz) {
        std::cerr << "ERROR: W component NNZ mismatch! Expected " << w_nnz << " but got " << c << std::endl;
    }

    for (int i = 0; i < w_nnz; ++i) {
        if (w_rows[i] < 0 || w_rows[i] >= w_dof_count || 
            w_collums[i] < 0 || w_collums[i] >= w_dof_count) {
            std::cerr << "ERROR W: Out of bounds at entry " << i 
                      << " - row=" << w_rows[i] 
                      << " col=" << w_collums[i] 
                      << " MatSize=" << w_dof_count << std::endl;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets_w;
    triplets_w.reserve(w_nnz);
    for (int i = 0; i < w_nnz; ++i)
    {
        triplets_w.push_back(Triplet(w_rows[i], w_collums[i], w_values[i]));
    }

    SparseMatrix mat_w(w_dof_count, w_dof_count);
    mat_w.setFromTriplets(triplets_w.begin(), triplets_w.end());
    SparseMatrix identity_w(w_dof_count, w_dof_count);
    identity_w.setIdentity();
    
    // implicfit difculsili mat
    DiffusionSolver3D::W_DIFFUSION_MATRIX_EIGEN = identity_w - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_w;

    free(w_collums);
    free(w_rows);
    free(w_values);  
};  

void DiffusionSolver3D::SolveDiffusion_U_Eigen(MAC* gridAnt,MAC* gridSol){
    DiffusionSolver3D::Update_U_DiffusionMatrix(gridAnt);
    double dh = gridAnt->dh;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);
    // ========== Solve U component ==========
    int u_matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();
    VectorXd RHS_U = VectorXd(u_matrix_size);
    RHS_U.setZero();
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k  = 1;k<Nz;k++){
                int id = Get_U_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= u_matrix_size) {
                    std::cerr << "ERROR: U RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_U(id) = gridAnt->GetU(i, j,k);
            }

        }
    }

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
    //copy it back
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 2; j < Nx; j++)
        {
            for(int k  = 1;k<Nz;k++){
                int id = Get_U_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= u_matrix_size) {
                    std::cerr << "ERROR: U solution index out of bounds! id=" << id   << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetU(i, j, k,SOL_U(id));

            }
        }
    }
};
void DiffusionSolver3D::SolveDiffusion_V_Eigen(MAC* gridAnt,MAC* gridSol){
    DiffusionSolver3D::Update_V_DiffusionMatrix(gridAnt);

// ========== Solve V component ==========
    double dh = gridAnt->dh;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);

    int v_matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();
    std::cout << "V matrix size: " << v_matrix_size << std::endl;
    VectorXd RHS_V = VectorXd(v_matrix_size);
    RHS_V.setZero();
    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int id = Get_V_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= v_matrix_size) {
                    std::cerr << "ERROR: V RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_V(id) = gridAnt->GetV(i, j,k);  // Previous timestep value
            }
        }
    }

    solver.compute(V_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }
    Eigen::VectorXd SOL_V = solver.solve(RHS_V);


    for (int i = 2; i < Ny; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 1;k<Nz-1;k++){
                int id = Get_V_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= v_matrix_size) {
                    std::cerr << "ERROR: V solution index out of bounds! id=" << id     << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetV(i, j,k, SOL_V(id));
            }
        }
        
    }


};
void DiffusionSolver3D::SolveDiffusion_W_Eigen(MAC* gridAnt,MAC* gridSol){
        DiffusionSolver3D::Update_W_DiffusionMatrix(gridAnt);

    //============= W Solve
    double dh = gridAnt->dh;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);
    int w_matrix_size = W_DIFFUSION_MATRIX_EIGEN.rows();
    std::cout << "W matrix size: " << w_matrix_size << std::endl;
    VectorXd RHS_W = VectorXd(w_matrix_size);
    RHS_W.setZero();
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int id = Get_W_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= w_matrix_size) {
                    std::cerr << "ERROR: W RHS index out of bounds! id=" << id 
                              << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                RHS_W(id) = gridAnt->GetW(i, j,k);  // Previous timestep value
            }
        }
    }

    solver.compute(W_DIFFUSION_MATRIX_EIGEN);
    
    if(solver.info() != Eigen::Success) {
        std::cerr << "ERROR: W matrix decomposition failed!" << std::endl;
        return;
    }
    Eigen::VectorXd SOL_W = solver.solve(RHS_W);


    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for(int k = 2;k<Nz;k++){
                int id = Get_W_IDP(i, j,k);
                if (id == -1) continue;
                if (id >= w_matrix_size) {
                    std::cerr << "ERROR: W solution index out of bounds! id=" << id     << " at (" << i << "," << j << "," << k << ")" << std::endl;
                    continue;
                }
                gridSol->SetW(i, j,k, SOL_W(id));
            }
        }
    }

    
};