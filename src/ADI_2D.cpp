#include "headers/Solvers/ADI_2D.h"
//X
MatrixXd ADI2D::U_X_Matrix = MatrixXd();
MatrixXd ADI2D::V_X_Matrix = MatrixXd();
MatrixXd ADI2D::W_X_Matrix = MatrixXd();
MatrixXd ADI2D::U_X_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::V_X_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::W_X_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::U_X_DIFF_Matrix = MatrixXd();
MatrixXd ADI2D::V_X_DIFF_Matrix = MatrixXd();
MatrixXd ADI2D::W_X_DIFF_Matrix = MatrixXd();

VectorXd ADI2D::U_X_Font = VectorXd();
VectorXd ADI2D::V_X_Font = VectorXd();
VectorXd ADI2D::W_X_Font = VectorXd();
VectorXd ADI2D::U_X_SOL = VectorXd();
VectorXd ADI2D::V_X_SOL = VectorXd();
VectorXd ADI2D::W_X_SOL = VectorXd();

//Y

MatrixXd ADI2D::U_Y_Matrix = MatrixXd();
MatrixXd ADI2D::V_Y_Matrix = MatrixXd();
MatrixXd ADI2D::W_Y_Matrix = MatrixXd();
MatrixXd ADI2D::U_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::V_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::W_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI2D::U_Y_DIFF_Matrix = MatrixXd();
MatrixXd ADI2D::V_Y_DIFF_Matrix = MatrixXd();
MatrixXd ADI2D::W_Y_DIFF_Matrix = MatrixXd();

VectorXd ADI2D::U_Y_Font = VectorXd();
VectorXd ADI2D::V_Y_Font = VectorXd();
VectorXd ADI2D::W_Y_Font = VectorXd();
VectorXd ADI2D::U_Y_SOL = VectorXd();
VectorXd ADI2D::V_Y_SOL = VectorXd();
VectorXd ADI2D::W_Y_SOL = VectorXd();



MAC ADI2D::X_STEP_SOL = MAC();
MAC ADI2D::Y_STEP_SOL = MAC();



double ADI2D::SIG = 0.0;

double ADI2D::dh = 0.0;



Vec2(*ADI2D::VelocityBorderFunction)(double, double,double) = nullptr;
Vec2(*ADI2D::VelocityFontFunction)(double, double,double) = nullptr;
double(*ADI2D::PressureFunction)(double, double,double) = nullptr;



//PARALLEL OPENMP TESTING


void ADI2D::SolveADI_X_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;

    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*2)) * (SIMULATION.EPS);
    double rho = (dt/(4*dh));


    
    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors, i am going clinically insane
        Eigen::MatrixXd local_U_X_CONV_Matrix = U_X_CONV_Matrix;
        Eigen::MatrixXd local_U_X_DIFF_Matrix = U_X_DIFF_Matrix;
        Eigen::VectorXd local_U_X_Font = U_X_Font;
        Eigen::VectorXd local_U_X_SOL;
        Eigen::MatrixXd local_U_X_Matrix;

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
                int c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    double f_ijk = ADI2D::VelocityFontFunction(j*dh, i*dh + dh/2.0, time).u;
                    //ADI2D::VelocityBorderFunction(j*dh, i*dh + dh/2.0, k*dh + dh/2.0, time).u;

                    double Y_DIFFUSION_TERM = (gridAnt->GetU(i+1,j) - 2*gridAnt->GetU(i,j) + gridAnt->GetU(i-1,j));
                    double V_CONVECTIVE_TERM = velocityField->getVatU(i,j)*((gridAnt->GetU(i+1,j) - gridAnt->GetU(i-1,j)));

                    local_U_X_Font(c) = gridAnt->GetU(i,j) + sig*(Y_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM) - (dt/2.0)*f_ijk;

                    if(c != local_U_X_CONV_Matrix.cols()-1 && c != 0) {
                        local_U_X_CONV_Matrix(c,c+1) = rho*velocityField->GetU(i,j);
                        local_U_X_CONV_Matrix(c,c-1) = -rho*velocityField->GetU(i,j);
                    }
                    c++;
                }

                local_U_X_CONV_Matrix(0,1) =      rho*gridAnt->GetU(i,2);
                local_U_X_CONV_Matrix(c-1,c-2) = -rho*gridAnt->GetU(i,Nx+1-3);

                local_U_X_Font(0)   +=  rho*gridSol->GetU(i,1)     *velocityField->GetU(i,2) +      sig*((gridSol->GetU(i,1)));
                local_U_X_Font(c-1) += -rho*gridSol->GetU(i,Nx+1-2)*velocityField->GetU(i,Nx+1-3) + sig*(gridSol->GetU(i,Nx+1-2));
                

                local_U_X_Matrix = local_U_X_CONV_Matrix + local_U_X_DIFF_Matrix;

                //SOLID MASK CHECKING
                //yes, it is inefficient, but honestly, we are already solving so fast it barely makes a difference
                //and this is the easiest way to implement this
                
                c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    if(gridSol->GetSolid(i,j) == SOLID_CELL || gridSol->GetSolid(i,j-1) == SOLID_CELL ){
                        local_U_X_Font(c) = 0.0;
                        local_U_X_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_U_X_Matrix(c,c-1) = 0.0; 
                            local_U_X_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_U_X_CONV_Matrix.cols()-1){
                            local_U_X_Matrix(c,c+1) = 0.0;
                            local_U_X_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }


                TDMA(local_U_X_Matrix,local_U_X_Font,local_U_X_SOL);




                c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    gridSol->SetU(i,j, local_U_X_SOL(c));
                    c++;
                }

                
            
        }
    }
}
void ADI2D::SolveADI_Y_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*2)) * (SIMULATION.EPS);
    double rho = (dt/(4*dh));
    
    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_U_Y_CONV_Matrix = U_Y_CONV_Matrix;
        Eigen::MatrixXd local_U_Y_DIFF_Matrix = U_Y_DIFF_Matrix;
        Eigen::VectorXd local_U_Y_Font = U_Y_Font;
        Eigen::VectorXd local_U_Y_SOL;
        Eigen::MatrixXd local_U_Y_Matrix;

        #pragma omp for
        for(int j = 2; j < Nx+1-2; j++) {
                int c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    double f_ijk = ADI2D::VelocityFontFunction(j*dh, i*dh + dh/2.0, time).u;
                    

                    double X_DIFFUSION_TERM = (gridAnt->GetU(i,j+1) - 2*gridAnt->GetU(i,j) + gridAnt->GetU(i,j-1));

                    double U_CONVECTIVE_TERM = velocityField->GetU(i,j)*((gridAnt->GetU(i,j+1) - gridAnt->GetU(i,j-1)));

                    
                    local_U_Y_Font(c) = gridAnt->GetU(i,j) + sig*(X_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM) - (dt/2.0)*f_ijk;

                    if(c != local_U_Y_CONV_Matrix.cols()-1 && c != 0) {
                        local_U_Y_CONV_Matrix(c,c+1) = rho*(velocityField->getVatU(i,j));
                        local_U_Y_CONV_Matrix(c,c-1) = -rho*(velocityField->getVatU(i,j));
                    }
                    c++;
                }

                local_U_Y_CONV_Matrix(0,1) =      rho*(velocityField->getVatU(1,j));
                local_U_Y_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getVatU(Ny-2,j));

                // Border terms
                local_U_Y_Font(0)   +=   rho*velocityField->getVatU(1,j)*gridSol->GetU(0,j)       + sig*((gridSol->GetU(0,j)));
                local_U_Y_Font(c-1) +=  -rho*velocityField->getVatU(Ny-2,j)*gridSol->GetU(Ny-1,j) + sig*((gridSol->GetU(Ny-1,j)));

                local_U_Y_Matrix = local_U_Y_CONV_Matrix + local_U_Y_DIFF_Matrix;

                c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    if(gridSol->GetSolid(i,j) == SOLID_CELL || gridSol->GetSolid(i,j-1) == SOLID_CELL){
                        local_U_Y_Font(c) = 0.0;
                        local_U_Y_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_U_Y_Matrix(c,c-1) = 0.0; 
                            local_U_Y_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_U_Y_CONV_Matrix.cols()-1){
                            local_U_Y_Matrix(c,c+1) = 0.0;
                            local_U_Y_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }


                TDMA(local_U_Y_Matrix,local_U_Y_Font,local_U_Y_SOL);

                c = 0;
                // Store the solution
                for(int i = 1; i < Ny-1; i++) {
                    gridSol->SetU(i,j, local_U_Y_SOL(c));
                    c++;
                }
            
        }
    }
}


void ADI2D::SolveADI_X_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*2)) * (SIMULATION.EPS);
    double rho = (dt/(4*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_V_X_CONV_Matrix = V_X_CONV_Matrix;
        Eigen::MatrixXd local_V_X_DIFF_Matrix = V_X_DIFF_Matrix;
        Eigen::VectorXd local_V_X_Font = V_X_Font;
        Eigen::VectorXd local_V_X_SOL;
        Eigen::MatrixXd local_V_X_Matrix;

        #pragma omp for
        for(int i = 2; i < Ny+1-2; i++) {
                int c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    double f_ijk = ADI2D::VelocityFontFunction(j*dh + dh/2.0, i*dh, time).v;

                    double Y_DIFFUSION_TERM = (gridAnt->GetV(i+1,j) - 2*gridAnt->GetV(i,j) + gridAnt->GetV(i-1,j));
                    double V_CONVECTIVE_TERM = (velocityField->GetV(i,j)) * ((gridAnt->GetV(i+1,j) - gridAnt->GetV(i-1,j)));

                    local_V_X_Font(c) = gridAnt->GetV(i,j) + sig*(Y_DIFFUSION_TERM ) - rho*(V_CONVECTIVE_TERM ) - (dt/2.0)*f_ijk;

                    if(c != local_V_X_CONV_Matrix.cols()-1 && c != 0) {
                        local_V_X_CONV_Matrix(c,c+1) = rho * velocityField->getUatV(i,j);
                        local_V_X_CONV_Matrix(c,c-1) = -rho * velocityField->getUatV(i,j);
                    }
                    c++;
                }

                // Set boundary elements
                local_V_X_CONV_Matrix(0,1) =      rho * velocityField->getUatV(i,1);
                local_V_X_CONV_Matrix(c-1,c-2) = -rho * velocityField->getUatV(i,Nx-2);

                // Apply boundary conditions
                local_V_X_Font(0)   +=  rho  * gridSol->GetV(i,0)    * velocityField->getUatV(i,1)    + sig * (gridSol->GetV(i,0));
                local_V_X_Font(c-1) += -rho * gridSol->GetV(i,Nx-1) * velocityField->getUatV(i,Nx-2) + sig * (gridSol->GetV(i,Nx-1));

                // Solve the system
                local_V_X_Matrix = local_V_X_CONV_Matrix + local_V_X_DIFF_Matrix;

                c = 0;
                for(int j = 1; j < Nx-1; j++)  {
                    if(gridSol->GetSolid(i,j) == SOLID_CELL || gridSol->GetSolid(i-1,j) == SOLID_CELL ){
                        local_V_X_Font(c) = 0.0;
                        local_V_X_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_V_X_Matrix(c,c-1) = 0.0; 
                            local_V_X_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_V_X_CONV_Matrix.cols()-1){
                            local_V_X_Matrix(c,c+1) = 0.0;
                            local_V_X_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }

                TDMA(local_V_X_Matrix,local_V_X_Font,local_V_X_SOL);
                // Store the solution
                c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    gridSol->SetV(i,j, local_V_X_SOL(c));
                    c++;
                }
            
        }
    }
}
void ADI2D::SolveADI_Y_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*2)) * (SIMULATION.EPS);
    double rho = (dt/(4*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_V_Y_CONV_Matrix = V_Y_CONV_Matrix;
        Eigen::MatrixXd local_V_Y_DIFF_Matrix = V_Y_DIFF_Matrix;
        Eigen::VectorXd local_V_Y_Font = V_Y_Font;
        Eigen::VectorXd local_V_Y_SOL;
        Eigen::MatrixXd local_V_Y_Matrix;

        #pragma omp for
        for(int j = 1; j < Nx-1; j++) {

                int c = 0;
                for(int i = 2; i < Ny+1-2; i++) {
                    double f_ijk = ADI2D::VelocityFontFunction(j*dh + dh/2.0, i*dh, time).v;

                    double X_DIFFUSION_TERM = (gridAnt->GetV(i,j+1) - 2*gridAnt->GetV(i,j) + gridAnt->GetV(i,j-1));


                    double U_CONVECTIVE_TERM = velocityField->getUatV(i,j) * ((gridAnt->GetV(i,j+1) - gridAnt->GetV(i,j-1)));


                    local_V_Y_Font(c) = gridAnt->GetV(i,j) + sig*(X_DIFFUSION_TERM ) - rho*(U_CONVECTIVE_TERM ) - (dt/2.0)*f_ijk;

                    if(c != local_V_Y_CONV_Matrix.cols()-1 && c != 0) {
                        //TAYLOR_GREEN_VORTEX_VELOCITY(j*dh + dh/2.0, i*dh, k*dh + dh/2.0, time).v;
                        local_V_Y_CONV_Matrix(c,c+1) =  rho * velocityField->GetV(i,j);
                        local_V_Y_CONV_Matrix(c,c-1) = -rho * velocityField->GetV(i,j);
                    }
                    c++;
                }

                // Set boundary elements
                local_V_Y_CONV_Matrix(0,1) = rho * velocityField->GetV(2,j);
                local_V_Y_CONV_Matrix(c-1,c-2) = -rho * velocityField->GetV(Ny+1-3,j);
                // Apply boundary conditions
                local_V_Y_Font(0)   +=   rho * gridSol->GetV(1,j)    * velocityField->GetV(2,j)    + sig * gridSol->GetV(1,j);
                local_V_Y_Font(c-1) +=  -rho * gridSol->GetV(Ny+1-2,j) * velocityField->GetV(Ny+1-3,j) + sig * gridSol->GetV(Ny+1-2,j);

                //if(j == 4 && k == 4){
                //    exportVectorToFile(local_V_Y_Font,"Exports/EigenVector/V_Y_FONT.csv");
                //    exportMatrixToFile(local_V_Y_Matrix,"Exports/EigenVector/V_Y_MATRIX.csv");
                //}

                // Solve the system
                local_V_Y_Matrix = local_V_Y_CONV_Matrix + local_V_Y_DIFF_Matrix;
                c = 0;
                for(int i = 2; i < Ny+1-2; i++)  {
                    if(gridSol->GetSolid(i,j) == SOLID_CELL || gridSol->GetSolid(i-1,j) == SOLID_CELL){
                        local_V_Y_Font(c) = 0.0;
                        local_V_Y_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_V_Y_Matrix(c,c-1) = 0.0; 
                            local_V_Y_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_V_Y_CONV_Matrix.cols()-1){
                            local_V_Y_Matrix(c,c+1) = 0.0;
                            local_V_Y_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }

                TDMA(local_V_Y_Matrix,local_V_Y_Font,local_V_Y_SOL);
                // Store the solution
                c = 0;
                for(int i = 2; i < Ny+1-2; i++) {
                    gridSol->SetV(i,j, local_V_Y_SOL(c));
                    c++;
                }
            
        }
    }
}






void ADI2D::InitializeADI2D(MAC* grid,double dt,Vec2(*VelocityBorderFunction)(double, double, double),Vec2(*VelocityFont)(double, double,double),double(*PressureFunction)(double, double, double)){
    int Nx = grid->Nx;
    int Ny = grid->Ny;


    ADI2D::VelocityBorderFunction = VelocityBorderFunction;
    ADI2D::VelocityFontFunction = VelocityFont;
    ADI2D::PressureFunction = PressureFunction;

    
    ADI2D::dh = grid->dh;
    
    //sigma coefficient
    ADI2D::SIG = (SIMULATION.dt/(dh*dh*2.0))* (SIMULATION.EPS);


    
    //please pay lots of attention here, since we have different ddirection and nodes, we have lots of matrixes and fonts, we pre allocated them to save time later
    //X DIR MATRIXES
    int u_size = (Nx + 1) - 4;   // U matrix size at x dir has less nodes, check diagram
    int v_size = (Nx)     - 2;   // V matrix size is normal

    
    ADI2D::U_X_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_X_Matrix = MatrixXd::Zero(v_size, v_size);

    ADI2D::U_X_CONV_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_X_CONV_Matrix = MatrixXd::Zero(v_size, v_size);


    ADI2D::U_X_DIFF_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_X_DIFF_Matrix = MatrixXd::Zero(v_size, v_size);


    ADI2D::U_X_Font = VectorXd::Zero(u_size);
    ADI2D::V_X_Font = VectorXd::Zero(v_size);


    ADI2D::U_X_SOL = VectorXd::Zero(u_size);
    ADI2D::V_X_SOL = VectorXd::Zero(v_size);

    
    // Set main diagonals
    //ADI2D::SIG = (dt / (grid->dh * 3.0)); past definition
    U_X_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, (2 * SIG) + 1);
    V_X_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, (2 * SIG)+ 1);


    
    // Set sub-diagonals (-SIG values), will also work for a 4x4 in one value bcause for is an if too
    for (int i = 1; i < u_size; i++) {
        U_X_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
        U_X_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
    }
    
    for (int i = 1; i < v_size; i++) {
        V_X_DIFF_Matrix(i, i-1) = -SIG;      
        V_X_DIFF_Matrix(i-1, i) = -SIG;    
    }




    // Y DIR MATRIXES
    u_size = (Ny )    - 2;   
    v_size = (Ny+1)   - 4;   


    ADI2D::U_Y_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_Y_Matrix = MatrixXd::Zero(v_size, v_size);


    ADI2D::U_Y_CONV_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_Y_CONV_Matrix = MatrixXd::Zero(v_size, v_size);


    ADI2D::U_Y_DIFF_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI2D::V_Y_DIFF_Matrix = MatrixXd::Zero(v_size, v_size);


    ADI2D::U_Y_Font=  VectorXd::Zero(u_size);
    ADI2D::V_Y_Font = VectorXd::Zero(v_size);


    ADI2D::U_Y_SOL=  VectorXd::Zero(u_size);
    ADI2D::V_Y_SOL = VectorXd::Zero(v_size);

    // Set main diagonals
    U_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, 2 * SIG + 1);
    V_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, 2 * SIG + 1);



    for (int i = 1; i < u_size; i++) {
        U_Y_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
        U_Y_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
    }
    
    for (int i = 1; i < v_size; i++) {
        V_Y_DIFF_Matrix(i, i-1) = -SIG;      
        V_Y_DIFF_Matrix(i-1, i) = -SIG;    
    }
    



    ADI2D::X_STEP_SOL.InitializeGrid(grid->omega,true);
    ADI2D::Y_STEP_SOL.InitializeGrid(grid->omega,true);


    ADI2D::X_STEP_SOL.CopyGrid(*grid);
    ADI2D::Y_STEP_SOL.CopyGrid(*grid);




}

//IM BREAKING the component on the exit
void ADI2D::SolveADIStep(MAC* gridAnt,MAC* gridSol,double time){

    ADI2D::SIG = (SIMULATION.dt/(dh*dh*2.0))* (SIMULATION.EPS);

    double start = omp_get_wtime();

    //we might as well just make the extra operations innsntead of making this garbage!
    if(ADAPTATIVE_TIMESTEP){
        int u_size = (gridAnt->Nx + 1) - 4;   // U matrix size at x dir has less nodes, check diagram
        int v_size = (gridAnt->Nx)     - 2;   // V matrix size is normal


        U_X_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, (2 * SIG) + 1);
        V_X_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, (2 * SIG)+ 1);
        // Set sub-diagonals (-SIG values), will also work for a 4x4 in one value bcause for is an if too
        for (int i = 1; i < u_size; i++) {
            U_X_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
            U_X_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
        }

        for (int i = 1; i < v_size; i++) {
            V_X_DIFF_Matrix(i, i-1) = -SIG;      
            V_X_DIFF_Matrix(i-1, i) = -SIG;    
        }


        u_size = (gridAnt->Ny )    - 2;   
        v_size = (gridAnt->Ny+1)   - 4;   


        U_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, 2 * SIG + 1);
        V_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, 2 * SIG + 1);
        for (int i = 1; i < u_size; i++) {
            U_Y_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
            U_Y_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
        }

        for (int i = 1; i < v_size; i++) {
            V_Y_DIFF_Matrix(i, i-1) = -SIG;      
            V_Y_DIFF_Matrix(i-1, i) = -SIG;    
        }



    }


 
    time += SIMULATION.dt/2.0;
    ADI2D::X_STEP_SOL.SetBorder(ADI2D::VelocityBorderFunction,ADI2D::PressureFunction,time);
    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE ) ADI2D::X_STEP_SOL.SetNeumannBorder();
    SolveADI_X_U_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);
    SolveADI_X_V_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);



    time += SIMULATION.dt/2.0;
    ADI2D::Y_STEP_SOL.SetBorder(ADI2D::VelocityBorderFunction,ADI2D::PressureFunction,time);
    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE ) ADI2D::Y_STEP_SOL.SetNeumannBorder();
    SolveADI_Y_U_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);
    SolveADI_Y_V_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);


    double end = omp_get_wtime();

    SIMULATION.lastADISolveTime = end - start;

    //WriteToCSV("Data/Times/time_adi2D_16.csv",end-start);

    //dont know if i need this!


    gridSol->CopyGrid(Y_STEP_SOL);


}