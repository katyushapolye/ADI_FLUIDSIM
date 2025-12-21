#include "headers/PressureSolver.h"

int PressureSolver::Nx = 0;
int PressureSolver::Ny = 0;
int PressureSolver::Nz = 0;
int PressureSolver::NON_ZERO = 0;
double PressureSolver::dt= 0;
VectorXd PressureSolver::IDP = VectorXd();

int *PressureSolver::collums = nullptr;
int *PressureSolver::rows = nullptr;
double *PressureSolver::values = nullptr;
CSRMatrix *PressureSolver::PRESSURE_MATRIX = nullptr;




std::vector<Tensor<double, 3>> PressureSolver::PRESSURE_MASK = std::vector<Tensor<double, 3>>();
SparseMatrix PressureSolver::PRESSURE_MATRIX_EIGEN;



// Expensive function, but it is only called once
void PressureSolver::InitializePressureSolver(MAC *grid,double dt)
{
    PressureSolver::Nx = SIMULATION.Nx;
    PressureSolver::Ny = SIMULATION.Ny;
    PressureSolver::Nz = SIMULATION.Nz;
    PressureSolver::PRESSURE_MASK.reserve(grid->GetFluidCellCount());
    PressureSolver::dt = dt;
    double dh = SIMULATION.dh;
    PressureSolver::IDP = VectorXd(Nx * Ny * Nz);
    PressureSolver::IDP.setConstant(-1);
    Eigen::setNbThreads(8);




    



    int c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3, 3);
                mask.setZero();
                
                // ADD THIS BOUNDARY CHECK (like 2D version)
                if (i == 0 || j == 0 || k == 0 || 
                    i == Ny-1 || j == Nx-1 || k == Nz-1)
                {
                    PressureSolver::PRESSURE_MASK.push_back(mask);
                    SetIDP(i, j, k, -1);
                    continue;
                }
                
                // Now check for solid/empty/inflow
                if (grid->GetSolid(i, j, k) == SOLID_CELL || 
                    grid->GetSolid(i,j,k) == EMPTY_CELL || 
                    grid->GetSolid(i,j,k) == INFLOW_CELL)
                {
                    PressureSolver::PRESSURE_MASK.push_back(mask);
                    SetIDP(i, j, k, -1);
                    continue;
                }
                
                // FLUID CELL - build stencil
                NON_ZERO++; // diagonal entry
                
                // Check fluid neighbors (add off-diagonal entries)
                if (grid->GetSolid(i + 1, j, k) == FLUID_CELL)
                {
                    mask(2, 1, 1) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                if (grid->GetSolid(i - 1, j, k) == FLUID_CELL)
                {
                    mask(0, 1, 1) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                if (grid->GetSolid(i, j + 1, k) == FLUID_CELL)
                {
                    mask(1, 2, 1) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                if (grid->GetSolid(i, j - 1, k) == FLUID_CELL)
                {
                    mask(1, 0, 1) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                if (grid->GetSolid(i, j, k + 1) == FLUID_CELL)
                {
                    mask(1, 1, 2) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                if (grid->GetSolid(i, j, k - 1) == FLUID_CELL)
                {
                    mask(1, 1, 0) = 1.0;
                    mask(1, 1, 1) -= 1.0;
                    NON_ZERO++;
                }
                
                // FIX 2: Check empty neighbors (modify diagonal only, NO NON_ZERO++)
                if (grid->GetSolid(i + 1, j, k) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                    // DO NOT increment NON_ZERO here!
                }
                if (grid->GetSolid(i - 1, j, k) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                }
                if (grid->GetSolid(i, j + 1, k) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                }
                if (grid->GetSolid(i, j - 1, k) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                }
                if (grid->GetSolid(i, j, k + 1) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                }
                if (grid->GetSolid(i, j, k - 1) == EMPTY_CELL)
                {
                    mask(1, 1, 1) -= 1.0;
                }
                
                SetIDP(i, j, k, c);
                c++;
                PressureSolver::PRESSURE_MASK.push_back(mask);
            }
        }
    }



    // create pressure matrix in COO representation, we later convert it to CSR,
    // ship to the GPU and so on
    // pay lot of attention on this function, it is complex!
    collums = (int *)malloc(sizeof(int) * NON_ZERO);
    rows = (int *)malloc(sizeof(int) * NON_ZERO);
    values = (double *)calloc(NON_ZERO,sizeof(double));
    int MatSize = grid->GetFluidCellCount();



    int MAT_LINE = 0;
    int MAT_COLLUM = 0;
    c = 0;

    for (int i = 1; i < Ny-1 ; i++)
    {
        for (int j = 1; j < Nx-1 ; j++)
        {
            for (int k = 1; k < Nz-1 ; k++)
            {
                Tensor<double, 3> mask = GetPressureMask(i, j, k);

            
                // get row index once
                int MAT_LINE = GetIDP(i, j, k);

                // if this cell is not a FLUID cell, skip entirely
                if (MAT_LINE == -1)
                    continue;

                // ------------------------------------------------------------
                // Diagonal (center)
                // ------------------------------------------------------------
                if (mask(1, 1, 1) != 0.0)
                {
                    rows[c]    = MAT_LINE;
                    collums[c] = MAT_LINE;
                    values[c] += mask(1, 1, 1);
                    c++;
                }

                // ------------------------------------------------------------
                // +X neighbor
                // ------------------------------------------------------------
                if (mask(2, 1, 1) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i + 1, j, k);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(2, 1, 1);
                        c++;
                    }
                }

                // ------------------------------------------------------------
                // -X neighbor
                // ------------------------------------------------------------
                if (mask(0, 1, 1) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i - 1, j, k);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(0, 1, 1);
                        c++;
                    }
                }

                // ------------------------------------------------------------
                // +Y neighbor
                // ------------------------------------------------------------
                if (mask(1, 2, 1) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i, j + 1, k);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(1, 2, 1);
                        c++;
                    }
                }

                // ------------------------------------------------------------
                // -Y neighbor
                // ------------------------------------------------------------
                if (mask(1, 0, 1) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i, j - 1, k);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(1, 0, 1);
                        c++;
                    }
                }

                // ------------------------------------------------------------
                // +Z neighbor
                // ------------------------------------------------------------
                if (mask(1, 1, 2) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i, j, k + 1);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(1, 1, 2);
                        c++;
                    }
                }

                // ------------------------------------------------------------
                // -Z neighbor
                // ------------------------------------------------------------
                if (mask(1, 1, 0) != 0.0)
                {
                    int MAT_COLLUM = GetIDP(i, j, k - 1);
                    if (MAT_COLLUM != -1)
                    {
                        rows[c]    = MAT_LINE;
                        collums[c] = MAT_COLLUM;
                        values[c] += mask(1, 1, 0);
                        c++;
                    }
                }


                //testing stuff, second order newmman boundary condition
  


            }
        }
    }




    // matrix should be finished here
    //std::cout << "Pressure Sparse Matrix is finished!\n";



    std::vector<Eigen::Triplet<double>> triplets;
    //triplets.reserve(NON_ZERO);
    

    for (int i = 0; i < NON_ZERO; ++i) {
        //printf("Row: %d  - Collum: %d - Value: %f\n",rows[i],collums[i],values[i]);
        triplets.push_back(Triplet(rows[i], collums[i], values[i]));
    }


    
    // Create the sparse matrix
    SparseMatrix mat(MatSize, MatSize);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat =  ((1.0)/(dh*dh)) * mat;

    



    PressureSolver::PRESSURE_MATRIX_EIGEN = mat;
    //writeSparseMatrixToFile(PRESSURE_MATRIX_EIGEN,"Exports/Eigen/PressureMat.txt");

    PRESSURE_MATRIX = coo_to_csr(rows,collums,values,NON_ZERO,MatSize,MatSize);


    free(collums);
    free(rows);
    free(values);

   

    //AMGX_config_create(&AMGX_Handle->config,
    //"config_version=2, "
    //"solver=BiCGST, "
    //"preconditioner=MULTICOLOR_GS, "
    //"monitor_residual=1, "
    //"store_res_history=0, "
    //"print_solve_stats=0, "
    //"convergence=ABSOLUTE, "
    //"tolerance=1e-12, "
    //"max_iters=1000, "
    //"norm=LMAX");


   




}

void PressureSolver::SolvePressure_EIGEN(MAC* grid){
    CPUTimer timer;
    timer.start();
     double dh = grid->dh;
    int MatSize = grid->GetFluidCellCount(); // number of FLUID cells

    VectorXd LHS = VectorXd(MatSize);
    
    LHS.setZero();
    int c = 0;
    double mean = 0.0;  // FIX: Initialize to zero!
    
    // Build RHS vector
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                int id = GetIDP(i,j,k);
                if (id == -1) continue;

                LHS(id) = (1.0/dt) * grid->GetDivergencyAt(i,j,k);
                mean += LHS(id);

                if(grid->GetSolid(i,j-1,k) == INFLOW_CELL)
                {
                    LHS(id) += -(1.0/dt) *
                        (grid->GetU(i,1,k) -
                         SIMULATION.VelocityBoundaryFunction(
                            dh, i*dh + dh/2.0, k*dh + dh/2.0, 0).u);
                }
            }
        }
    }

    if(SIMULATION.NEEDS_COMPATIBILITY_CONDITION){
    //CONDICAO DED COMPATIBILIDADE - NAO SEI PORQUE
    mean = mean/MatSize;
    for(int i = 0;i< MatSize;i++){
     
       LHS(i) = LHS(i) - mean;
    }
    }


    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-12);
    solver.compute(PRESSURE_MATRIX_EIGEN);
    Eigen::VectorXd SOL = VectorXd(MatSize);
    SOL =  solver.solve(LHS);

    c = 0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                int id = GetIDP(i,j,k);
                if (id == -1) continue;
                grid->SetP(i,j,k, SOL(id));
            }
        }
    }

    grid->SetNeumannBorderPressure();

        double end = GetWallTime();

    SIMULATION.lastPressureSolveTime = timer.stop();

    




}


void PressureSolver::SolvePressure_AMGX(MAC* grid){
CPUTimer timer;
    timer.start();
    double dh = grid->dh;
    int MatSize = grid->GetFluidCellCount();
    double* LHS = (double*)calloc(MatSize,sizeof(double));
    double* SOL = (double*)calloc(MatSize,sizeof(double));



    int c = 0;
    double mean = 0;


    //ops!
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                if(GetIDP(i,j,k) != -1){
                //imposing boundary conditioons the SHITTTIEST way, JUST FOR TESTING!
                //LHS(c) =  (1.0/dt)*grid.GetDivergencyAt(i,j,k);
                LHS[GetIDP(i,j,k)] =  (1.0/dt)*grid->GetDivergencyAt(i,j,k); //for SPD
                mean += LHS[GetIDP(i,j,k)];


                //supprots inflow oon the -x to +x dir
                if(grid->GetSolid(i,j-1,k) == INFLOW_CELL){
                    LHS[GetIDP(i,j,k)] += -(1.0/dt)*( grid->GetU(i,1,k)-SIMULATION.VelocityBoundaryFunction(1*dh,i*dh + dh/2.0,k*dh +dh/2.0,0).u );//amayybe minus
                   
                }
                }
            
                //test BC
                //if(j == 1 && i > Ny/2){
                //    LHS(c) += -(1.0/dt)*( grid.GetU(i,1,k)-BACKWARDS_FACING_STEP(1*dh,i*dh + dh/2.0,k*dh +dh/2.0,0).u  );//amayybe minus
                //}
                
               // if(k == 1){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     //std::cout <<  LHS(c) << std::endl;
               //     LHS(c) +=  -(1.0/(dh*dh)) *grid.GetP(i,j,0);
//
               // }
//
               // if(k == Nz-2){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     LHS(c) +=   -(1.0/(dh*dh))*grid.GetP(i,j,Nz-1);
//
               // }
//
               // if(i == 1){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     LHS(c) +=   -(1.0/(dh*dh))*grid.GetP(0,j,k);
//
               // }
//
               // if(i == Ny-2){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     LHS(c) +=   -(1.0/(dh*dh))*grid.GetP(Ny-1,j,k);
//
               // }
//
               // if(j ==1){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     LHS(c) +=   -(1.0/(dh*dh))*grid.GetP(i,0,k);
//
               // }
//
               // if(j == Nx-2){
               //     //the term on the back goes to the font, since we multiply both sides foor -1 for conditioning, it goes positive
               //     LHS(c) +=   -(1.0/(dh*dh))*grid.GetP(i,Nx-1,k);
//
               // }




            }
            
        }
    }


    if(SIMULATION.NEEDS_COMPATIBILITY_CONDITION){
    //CONDICAO DED COMPATIBILIDADE - NAO SEI PORQUE
    mean = mean/MatSize;
    for(int i = 0;i< MatSize;i++){
     
       LHS[i] = LHS[i] - mean;
    }
    }









    //as crazy as it soundds, the error might be on the pressure transfer when there is a solid in the middle of the canal
    c = 0;

    for (int i = 1; i < Ny - 1; i++) {
        for (int j = 1; j < Nx - 1; j++) {
            for (int k = 1; k < Nz - 1; k++) {
                int id = GetIDP(i,j,k);
                if (id == -1) continue;
                grid->SetP(i,j,k, SOL[id]);
            }
        }
    }


    grid->SetNeumannBorderPressure();

    

    double end = GetWallTime();

    SIMULATION.lastPressureSolveTime = timer.stop();



    free(LHS);
    free(SOL);



}


void PressureSolver::ProjectPressure(MAC* grid){
    
    for (int i = 1; i < Ny-1 ; i++)
    {                  
        for (int j = 1; j < Nx +1-1; j++)
        {
            for (int k = 1; k < Nz-1; k++) 
            {

                //if the cell we are is not a solid wall, then we update
                if( !(grid->GetSolid(i,j,k) == SOLID_CELL || grid->GetSolid(i,j-1,k)==SOLID_CELL)){
                    grid->SetU(i,j,k, 
                        grid->GetU(i,j,k) - (dt) * grid->GetGradPxAt(i,j,k)
                        );
                    }
                
            }
        }
    }


    for (int i = 1; i < Ny+1-1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for (int k = 1; k < Nz-1; k++)
            {   if( !(grid->GetSolid(i,j,k) == SOLID_CELL || grid->GetSolid(i-1,j,k)==SOLID_CELL)){
                    grid->SetV(i,j,k, 
                    grid->GetV(i,j,k) - (dt) * grid->GetGradPyAt(i,j,k)

                );
            }
            


            }
        }
    }

    for (int i = 1; i < Ny-1; i++)
    {
        for (int j = 1; j < Nx-1; j++)
        {
            for (int k = 1; k < Nz+1-1; k++)
            {   
                if( !(grid->GetSolid(i,j,k) == SOLID_CELL || grid->GetSolid(i,j,k-1)==SOLID_CELL)){
                    grid->SetW(i,j,k, 
                    grid->GetW(i,j,k) - (dt) * grid->GetGradPzAt(i,j,k)

                );
            }
            


            }
        }
    }
    

    //only update if it needs update
    /*
    for (int i = 0; i < Ny ; i++)
    {                  
        for (int j = 0; j < Nx + 1; j++)
        {
            for (int k = 0; k < Nz; k++) 
            {
                if(grid->GetU_Update_Mask(i,j,k) == FLUID_CELL || grid->GetU_Update_Mask(i,j,k) == EMPTY_CELL){
                    grid->SetU(i,j,k, 
                    grid->GetU(i,j,k) - (dt) * grid->GetGradPxAt(i,j,k)
                    );
                }
                else{
                continue;
                }

            }
        }
    }


    for (int i = 0; i < Ny+1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                if(grid->GetV_Update_Mask(i,j,k) == FLUID_CELL || grid->GetV_Update_Mask(i,j,k) == EMPTY_CELL){
                    grid->SetV(i,j,k, 
                    grid->GetV(i,j,k) - (dt) * grid->GetGradPyAt(i,j,k)

                );
            }


            }
        }
    }

    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz+1; k++)
            {
                if(grid->GetW_Update_Mask(i,j,k) == FLUID_CELL || grid->GetW_Update_Mask(i,j,k) == EMPTY_CELL){
                    grid->SetW(i,j,k, 
                    grid->GetW(i,j,k) - (dt) * grid->GetGradPzAt(i,j,k)

                );
            }


            }
        }
    }
        */

    




}



