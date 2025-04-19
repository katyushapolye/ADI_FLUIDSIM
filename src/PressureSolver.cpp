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
AMGXSolver *PressureSolver::AMGX_Handle = nullptr;



std::vector<Tensor<double, 3>> PressureSolver::PRESSURE_MASK = std::vector<Tensor<double, 3>>();
SparseMatrix PressureSolver::PRESSURE_MATRIX_EIGEN;



// Expensive function, but it is only called once
void PressureSolver::InitializePressureSolver(MAC *grid,double dt)
{
    PressureSolver::Nx = SIMULATION.Nx;
    PressureSolver::Ny = SIMULATION.Ny;
    PressureSolver::Nz = SIMULATION.Nz;
    PressureSolver::PRESSURE_MASK.reserve(Nx * Ny * Nz);
    PressureSolver::dt = dt;
    double dh = SIMULATION.dh;
    PressureSolver::IDP = VectorXd(Nx * Ny * Nz);

    int c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                // add the pressure mask
                Tensor<double, 3> mask = Tensor<double, 3>(3, 3, 3);
                mask.setZero();
                if (grid->GetSolid(i, j, k) == SOLID_CELL || grid->GetSolid(i,j,k) == EMPTY_CELL || grid->GetSolid(i,j,k) == INFLOW_CELL)
                {
                    
                    PressureSolver::PRESSURE_MASK.push_back(mask); // all zero mask
                    SetIDP(i, j, k, -1);
                    continue;

                }
                else
                {
                    NON_ZERO++; // the cell we are is not solid, this will count the
                    // amount of non zero elements in the matrix

                    // if not on a solid wall, we check all directions of the stencil, and
                    // update the indexing grid

                    if (grid->GetSolid(i + 1, j, k) == FLUID_CELL)
                    {
                        mask(2, 1, 1) = 1.0;  // top
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;
                    }

                    if (grid->GetSolid(i - 1, j, k) == FLUID_CELL)
                    {
                        mask(0, 1, 1) = 1.0;  // botton
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j + 1, k) == FLUID_CELL)
                    {
                        mask(1, 2, 1) = 1.0;  // left
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j - 1, k) == FLUID_CELL)
                    {
                        mask(1, 0, 1) = 1.0;  // right
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j, k + 1) == FLUID_CELL)
                    {
                        mask(1, 1, 2) = 1.0;  // foward
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j, k - 1) == FLUID_CELL)
                    {
                        mask(1, 1, 0) = 1.0;  // foward
                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }


                    

                    if (grid->GetSolid(i + 1, j, k) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;
                    }

                    if (grid->GetSolid(i - 1, j, k) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j + 1, k) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j - 1, k) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j, k + 1) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }

                    if (grid->GetSolid(i, j, k - 1) == EMPTY_CELL)
                    {

                        mask(1, 1, 1) -= 1.0; // center
                        NON_ZERO++;           // the cell we are is not solid
                    }









                    SetIDP(i, j, k, c);
                    c++;
                    PressureSolver::PRESSURE_MASK.push_back(mask);
                }
            }
        }
    }




    // create pressure matrix in COO representation, we later convert it to CSR,
    // ship to the GPU and so on
    // pay lot of attention on this function, it is complex!
    collums = (int *)malloc(sizeof(int) * NON_ZERO);
    rows = (int *)malloc(sizeof(int) * NON_ZERO);
    values = (double *)calloc(NON_ZERO,sizeof(double));
    int MatSize = (Nx - 2) * (Ny - 2) * (Nz - 2);



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

            
                // if we are not on a empty mask (we could also check idp for -1!)
                if (mask(1, 1, 1) != 0)
                {
                    MAT_LINE = GetIDP(i, j, k);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_LINE;
                    values[c] += mask(1, 1, 1); // get the center of the stencil
                    c++;
                }

                // now onto each collum of this row, which are the neighboors
                if (mask(2, 1, 1) != 0.0)
                {
                    MAT_COLLUM = GetIDP(i + 1, j, k);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] +=  mask(2, 1, 1);
                    c++;
                }

                if (mask(0, 1, 1) != 0.0)
                {
                    MAT_COLLUM = GetIDP(i - 1, j, k);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(0, 1, 1);
                    c++;
                }

                if (mask(1, 2, 1) != 0.0)
                {

                    MAT_COLLUM = GetIDP(i, j + 1, k);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 2, 1);
                    c++;

                }

                if (mask(1, 0, 1) != 0.0)
                {
                    MAT_COLLUM = GetIDP(i, j - 1, k);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 0, 1);
                    c++;
                }

                if (mask(1, 1, 2) != 0.0)
                {
                    MAT_COLLUM = GetIDP(i, j, k + 1);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 1, 2);
                    c++;
                }

                if (mask(1, 1, 0) != 0.0)
                {
                    MAT_COLLUM = GetIDP(i, j, k - 1);
                    rows[c] = MAT_LINE;
                    collums[c] = MAT_COLLUM;
                    values[c] += mask(1, 1, 0);
                    c++;
                }

                //testing stuff, second order newmman boundary condition
  


            }
        }
    }




    // matrix should be finished here
    std::cout << "Pressure Sparse Matrix is finished!\n";



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
    writeSparseMatrixToFile(PRESSURE_MATRIX_EIGEN,"Exports/Eigen/PressureMat.txt");

    PRESSURE_MATRIX = coo_to_csr(rows,collums,values,NON_ZERO,MatSize,MatSize);

    free(collums);
    free(rows);
    free(values);
    AMGX_initialize();

    PressureSolver::AMGX_Handle = new AMGXSolver();


    const char* config = "config_version=2, solver=BICGSTAB, preconditioner=MULTICOLOR_GS, "
    "monitor_residual=1, store_res_history=1, "
    "convergence=ABSOLUTE, tolerance=1e-12, " //changed tol to less to see more pertubations
    "max_iters=5000, norm=LMAX";
    AMGX_config_create_from_file(&AMGX_Handle->config,"solver_config.txt");
    //AMGX_config_create(&AMGX_Handle->config, config);
    AMGX_resources_create_simple(&AMGX_Handle->rsrc,AMGX_Handle->config);

    //Creating AMGX handles check for the negative thing
                                                            //run on device (GPU), double precision for all, integer or indexes
    AMGX_matrix_create(&AMGX_Handle->AmgxA,AMGX_Handle->rsrc,AMGX_mode_dDDI);
    AMGX_vector_create(&AMGX_Handle->Amgxb,AMGX_Handle->rsrc,AMGX_mode_dDDI);
    AMGX_vector_create(&AMGX_Handle->Amgxx,AMGX_Handle->rsrc,AMGX_mode_dDDI);
    for(int i=0;i<NON_ZERO;i++){
        PRESSURE_MATRIX->values[i] = PRESSURE_MATRIX->values[i]* (1.0)/(dh*dh);
    }

    //the amgx matrix is not populated with the coeficient 1/dh**2
    AMGX_matrix_upload_all(AMGX_Handle->AmgxA,MatSize,NON_ZERO,1,1,PRESSURE_MATRIX->row_ptr,PRESSURE_MATRIX->col_ind,PRESSURE_MATRIX->values,nullptr);
    AMGX_vector_set_zero(AMGX_Handle->Amgxx,MatSize,1);
    AMGX_vector_set_zero(AMGX_Handle->Amgxb,MatSize,1);

    //in theory, we have a valid solvver here!
    AMGX_solver_create(&AMGX_Handle->solver,AMGX_Handle->rsrc,AMGX_mode_dDDI,AMGX_Handle->config);
    AMGX_solver_setup(AMGX_Handle->solver,AMGX_Handle->AmgxA);

    //PrintAMGXMatrix(AMGX_Handle,MatSize);





}

void PressureSolver::SolvePressure_EIGEN(MAC* grid){
    double dh = grid->dh;
    int MatSize = (Nx - 2) * (Ny - 2) * (Nz - 2);
    VectorXd LHS = VectorXd(MatSize);
    
    LHS.setZero();


    int c = 0;

    double mean = 0;



    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                //imposing boundary conditioons the SHITTTIEST way, JUST FOR TESTING!
                //LHS(c) =  (1.0/dt)*grid.GetDivergencyAt(i,j,k);
                LHS(c) =  (1.0/dt)*grid->GetDivergencyAt(i,j,k); //for SPD

                mean += LHS(c);


                //supprots inflow oon the -x to +x dir
                if(grid->GetSolid(i,j-1,k) == INFLOW_CELL){
                    LHS(c) += -(1.0/dt)*( grid->GetU(i,1,k)-SIMULATION.VelocityBoundaryFunction(1*dh,i*dh + dh/2.0,k*dh +dh/2.0,0).u );//amayybe minus
                }
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


                c++;
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
                grid->SetP(i,j,k,SOL(c));
                c++;
            }
        }
    }

    grid->SetNeumannBorderPressure();

    




}


void PressureSolver::SolvePressure_AMGX(MAC* grid){
    CPUTimer timer;
    timer.start();
    double dh = grid->dh;
    int MatSize = (Nx - 2) * (Ny - 2) * (Nz - 2);
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



                c++;
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


    //solving!
    AMGX_vector_upload(AMGX_Handle->Amgxb,MatSize,1,LHS);
    AMGX_solver_solve_with_0_initial_guess(AMGX_Handle->solver,AMGX_Handle->Amgxb,AMGX_Handle->Amgxx);
    AMGX_vector_download(AMGX_Handle->Amgxx,SOL);






    //as crazy as it soundds, the error might be on the pressure transfer when there is a solid in the middle of the canal
    c = 0;

    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                if(GetIDP(i,j,k) != -1){
                //if(grid->GetSolid(i,j,k) ==  FLUID_CELL){
                    grid->SetP(i,j,k,SOL[c]);
                    c++;
                }

            }
        }
    }

    grid->SetNeumannBorderPressure();

    

    double end = GetWallTime();

    SIMULATION.lastPressureSolveTime = timer.stop();



    free(LHS);
    free(SOL);
    AMGX_vector_set_zero(AMGX_Handle->Amgxx,MatSize,1);
    AMGX_vector_set_zero(AMGX_Handle->Amgxb,MatSize,1);



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




int PressureSolver::GetSolverIterations(){
    int iterations;
    AMGX_solver_get_iterations_number(AMGX_Handle->solver, &iterations);
    return iterations;
}