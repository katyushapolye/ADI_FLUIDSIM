#include "headers/PressureSolver_2D.h"

int PressureSolver2D::Nx = 0;
int PressureSolver2D::Ny = 0;
int PressureSolver2D::NON_ZERO = 0;
double PressureSolver2D::dt= 0;
VectorXd PressureSolver2D::IDP = VectorXd();

int *PressureSolver2D::collums = nullptr;
int *PressureSolver2D::rows = nullptr;
double *PressureSolver2D::values = nullptr;
CSRMatrix *PressureSolver2D::PRESSURE_MATRIX = nullptr;




std::vector<Tensor<double, 2>> PressureSolver2D::PRESSURE_MASK = std::vector<Tensor<double, 2>>();
SparseMatrix PressureSolver2D::PRESSURE_MATRIX_EIGEN;



void PressureSolver2D::InitializePressureSolver(MAC2D *grid, double dt)
{
    PressureSolver2D::Nx = SIMULATION2D.Nx;
    PressureSolver2D::Ny = SIMULATION2D.Ny;
    PressureSolver2D::PRESSURE_MASK.reserve(grid->GetFluidCellCount());
    PressureSolver2D::dt = dt;
    double dh = SIMULATION2D.dh;
    PressureSolver2D::IDP = VectorXd(Nx * Ny);
    PressureSolver2D::IDP.setConstant(-1);  // FIX 1: Initialize to -1
    Eigen::setNbThreads(8);

    int c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            Tensor<double, 2> mask = Tensor<double, 2>(3, 3);
            mask.setZero();
            
            // FIX 2: ADD BOUNDARY CHECK (like 3D version)
            if (i == 0 || j == 0 || i == Ny-1 || j == Nx-1)
            {
                PressureSolver2D::PRESSURE_MASK.push_back(mask);
                SetIDP(i, j, -1);
                continue;
            }
            
            // Now check for solid/empty/inflow
            if (grid->GetSolid(i, j) == SOLID_CELL || 
                grid->GetSolid(i, j) == EMPTY_CELL || 
                grid->GetSolid(i, j) == INFLOW_CELL)
            {
                PressureSolver2D::PRESSURE_MASK.push_back(mask);
                SetIDP(i, j, -1);
                continue;
            }
            
            // FLUID CELL - build stencil
            NON_ZERO++; // diagonal entry
            
            // Check fluid neighbors (add off-diagonal entries)
            if (grid->GetSolid(i + 1, j) == FLUID_CELL)
            {
                mask(2, 1) = 1.0;
                mask(1, 1) -= 1.0;
                NON_ZERO++;
            }
            if (grid->GetSolid(i - 1, j) == FLUID_CELL)
            {
                mask(0, 1) = 1.0;
                mask(1, 1) -= 1.0;
                NON_ZERO++;
            }
            if (grid->GetSolid(i, j + 1) == FLUID_CELL)
            {
                mask(1, 2) = 1.0;
                mask(1, 1) -= 1.0;
                NON_ZERO++;
            }
            if (grid->GetSolid(i, j - 1) == FLUID_CELL)
            {
                mask(1, 0) = 1.0;
                mask(1, 1) -= 1.0;
                NON_ZERO++;
            }
            
            // FIX 3: Check empty neighbors (modify diagonal only, NO NON_ZERO++)
            if (grid->GetSolid(i + 1, j) == EMPTY_CELL)
            {
                mask(1, 1) -= 1.0;
                // DO NOT increment NON_ZERO here!
            }
            if (grid->GetSolid(i - 1, j) == EMPTY_CELL)
            {
                mask(1, 1) -= 1.0;
            }
            if (grid->GetSolid(i, j + 1) == EMPTY_CELL)
            {
                mask(1, 1) -= 1.0;
            }
            if (grid->GetSolid(i, j - 1) == EMPTY_CELL)
            {
                mask(1, 1) -= 1.0;
            }
            
            SetIDP(i, j, c);
            c++;
            PressureSolver2D::PRESSURE_MASK.push_back(mask);
        }
    }

    // FIX 4: Use actual fluid cell count, not (Nx-2)*(Ny-2)
    collums = (int *)malloc(sizeof(int) * NON_ZERO);
    rows = (int *)malloc(sizeof(int) * NON_ZERO);
    values = (double *)calloc(NON_ZERO, sizeof(double));
    int MatSize = grid->GetFluidCellCount();

    c = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            Tensor<double, 2> mask = GetPressureMask(i, j);

            // Get row index once
            int MAT_LINE = GetIDP(i, j);
            
            // If this cell is not a FLUID cell, skip entirely
            if (MAT_LINE == -1)
                continue;

            // Diagonal (center)
            if (mask(1, 1) != 0.0)
            {
                rows[c] = MAT_LINE;
                collums[c] = MAT_LINE;
                values[c] += mask(1, 1);
                c++;
            }

            // +Y neighbor
            if (mask(2, 1) != 0.0)
            {
                int MAT_COLLUM = GetIDP(i + 1, j);
                if (MAT_COLLUM != -1)
                {
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(2, 1);
                    c++;
                }
            }

            // -Y neighbor
            if (mask(0, 1) != 0.0)
            {
                int MAT_COLLUM = GetIDP(i - 1, j);
                if (MAT_COLLUM != -1)
                {
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(0, 1);
                    c++;
                }
            }

            // +X neighbor
            if (mask(1, 2) != 0.0)
            {
                int MAT_COLLUM = GetIDP(i, j + 1);
                if (MAT_COLLUM != -1)
                {
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 2);
                    c++;
                }
            }

            // -X neighbor
            if (mask(1, 0) != 0.0)
            {
                int MAT_COLLUM = GetIDP(i, j - 1);
                if (MAT_COLLUM != -1)
                {
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 0);
                    c++;
                }
            }
        }
    }

    std::cout << "Pressure Sparse Matrix is finished!\n";

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < NON_ZERO; ++i)
    {
        triplets.push_back(Triplet(rows[i], collums[i], values[i]));
    }

    SparseMatrix mat(MatSize, MatSize);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat = ((1.0) / (dh * dh)) * mat;

    PressureSolver2D::PRESSURE_MATRIX_EIGEN = mat;

    PRESSURE_MATRIX = coo_to_csr(rows, collums, values, NON_ZERO, MatSize, MatSize);

    free(collums);
    free(rows);
    free(values);

    
}

void PressureSolver2D::SolvePressure_EIGEN(MAC2D* grid)
{

    PressureSolver2D::dt = SIMULATION2D.dt;  // FIX: Update dt!
    CPUTimer timer;
    timer.start();
    double dh = grid->dh;
    int MatSize = grid->GetFluidCellCount();
    
    VectorXd LHS = VectorXd(MatSize);
    LHS.setZero();
    
    double mean = 0.0;

    // Build RHS vector using IDP indexing
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            int id = GetIDP(i, j);
            if (id == -1) continue;

            LHS(id) = (1.0/dt) * grid->GetDivergencyAt(i, j);
            mean += LHS(id);

            if (grid->GetSolid(i, j-1) == INFLOW_CELL)
            {
                LHS(id) += -(1.0/dt) * 
                    (grid->GetU(i, 1) - 
                     SIMULATION2D.VelocityBoundaryFunction(dh, i*dh + dh/2.0, 0).u);
            }
        }
    }

    if (SIMULATION2D.NEEDS_COMPATIBILITY_CONDITION)  // FIX: Use SIMULATION2D
    {
        mean = mean / MatSize;
        for (int i = 0; i < MatSize; i++)
        {
            LHS(i) = LHS(i) - mean;
        }
    }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-9);
    solver.compute(PRESSURE_MATRIX_EIGEN);
    Eigen::VectorXd SOL = VectorXd(MatSize);
    SOL = solver.solve(LHS);

    // Write solution back using IDP indexing
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            int id = GetIDP(i, j);
            if (id == -1) continue;
            grid->SetP(i, j, SOL(id));
        }
    }

    grid->SetNeumannBorderPressure();
    SIMULATION2D.lastPressureSolveTime = timer.stop();  // FIX: Use SIMULATION2D
}


void PressureSolver2D::SolvePressure_AMGX(MAC2D* grid)
{
   
}


void PressureSolver2D::ProjectPressure(MAC2D* grid)
{
    // Project U velocities
    for (int i = 1; i < Ny-1; i++)
    {                  
        for (int j = 1; j < Nx+1-1; j++)
        {
            // If the cell we are in is not a solid wall, then we update
            if (!(grid->GetSolid(i, j) == SOLID_CELL || grid->GetSolid(i, j-1) == SOLID_CELL))
            {
                grid->SetU(i, j, 
                    grid->GetU(i, j) - dt * grid->GetGradPxAt(i, j)
                );
            }
        }
    }

    // Project V velocities
    for (int i = 1; i < Ny+1-1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            if (!(grid->GetSolid(i, j) == SOLID_CELL || grid->GetSolid(i-1, j) == SOLID_CELL))
            {
                grid->SetV(i, j, 
                    grid->GetV(i, j) - dt * grid->GetGradPyAt(i, j)
                );
            }
        }
    }
}




int PressureSolver2D::GetSolverIterations(){
    int iterations;
    return iterations;
}