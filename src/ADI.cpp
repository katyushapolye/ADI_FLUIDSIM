#include "headers/ADI.h"
//X
MatrixXd ADI::U_X_Matrix = MatrixXd();
MatrixXd ADI::V_X_Matrix = MatrixXd();
MatrixXd ADI::W_X_Matrix = MatrixXd();
MatrixXd ADI::U_X_CONV_Matrix = MatrixXd();
MatrixXd ADI::V_X_CONV_Matrix = MatrixXd();
MatrixXd ADI::W_X_CONV_Matrix = MatrixXd();
MatrixXd ADI::U_X_DIFF_Matrix = MatrixXd();
MatrixXd ADI::V_X_DIFF_Matrix = MatrixXd();
MatrixXd ADI::W_X_DIFF_Matrix = MatrixXd();

VectorXd ADI::U_X_Font = VectorXd();
VectorXd ADI::V_X_Font = VectorXd();
VectorXd ADI::W_X_Font = VectorXd();
VectorXd ADI::U_X_SOL = VectorXd();
VectorXd ADI::V_X_SOL = VectorXd();
VectorXd ADI::W_X_SOL = VectorXd();

//Y

MatrixXd ADI::U_Y_Matrix = MatrixXd();
MatrixXd ADI::V_Y_Matrix = MatrixXd();
MatrixXd ADI::W_Y_Matrix = MatrixXd();
MatrixXd ADI::U_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI::V_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI::W_Y_CONV_Matrix = MatrixXd();
MatrixXd ADI::U_Y_DIFF_Matrix = MatrixXd();
MatrixXd ADI::V_Y_DIFF_Matrix = MatrixXd();
MatrixXd ADI::W_Y_DIFF_Matrix = MatrixXd();

VectorXd ADI::U_Y_Font = VectorXd();
VectorXd ADI::V_Y_Font = VectorXd();
VectorXd ADI::W_Y_Font = VectorXd();
VectorXd ADI::U_Y_SOL = VectorXd();
VectorXd ADI::V_Y_SOL = VectorXd();
VectorXd ADI::W_Y_SOL = VectorXd();

//Z
MatrixXd ADI::U_Z_Matrix = MatrixXd();
MatrixXd ADI::V_Z_Matrix = MatrixXd();
MatrixXd ADI::W_Z_Matrix = MatrixXd();
MatrixXd ADI::U_Z_CONV_Matrix = MatrixXd();
MatrixXd ADI::V_Z_CONV_Matrix = MatrixXd();
MatrixXd ADI::W_Z_CONV_Matrix = MatrixXd();
MatrixXd ADI::U_Z_DIFF_Matrix = MatrixXd();
MatrixXd ADI::V_Z_DIFF_Matrix = MatrixXd();
MatrixXd ADI::W_Z_DIFF_Matrix = MatrixXd();

VectorXd ADI::U_Z_Font = VectorXd();
VectorXd ADI::V_Z_Font = VectorXd();
VectorXd ADI::W_Z_Font = VectorXd();
VectorXd ADI::U_Z_SOL = VectorXd();
VectorXd ADI::V_Z_SOL = VectorXd();
VectorXd ADI::W_Z_SOL = VectorXd();


MAC ADI::X_STEP_SOL = MAC();
MAC ADI::Y_STEP_SOL = MAC();
MAC ADI::Z_STEP_SOL = MAC();


double ADI::SIG = 0.0;
double ADI::dt = 0.0;
double ADI::dh = 0.0;



Vec3(*ADI::VelocityBorderFunction)(double, double, double,double) = nullptr;
Vec3(*ADI::VelocityFontFunction)(double, double, double,double) = nullptr;
double(*ADI::PressureFunction)(double, double, double,double) = nullptr;

/*
void ADI::SolveADI_X_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int i =1;i<Ny-1;i++){ //not considering the border terms on i
        for(int k = 1;k<Nz-1;k++){ //not considerng the border terms on k
            int c = 0;
            for(int j = 2;j<Nx+1-2;j++){ //loooping from j = 2 to Nx+1-2 because we have a MAC grid and aditional U components on the faces
                double f_ijk = ADI::VelocityFontFunction(j*dh,i*dh + dh/2.0,k*dh + dh/2.0,time).u; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));
                                            //This is the exact solution
                double V_CONVECTIVE_TERM = (velocityField->getVatU(i,j,k))*((gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k)));
                double W_CONVECTIVE_TERM = (velocityField->getWatU(i,j,k))*((gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1)));

                U_X_Font(c) = gridAnt->GetU(i,j,k) + sig*(Y_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != U_X_CONV_Matrix.cols()-1 && c != 0){
                    U_X_CONV_Matrix(c,c+1) =  rho*velocityField->GetU(i,j,k); //upper diag
                    U_X_CONV_Matrix(c,c-1) = -rho*velocityField->GetU(i,j,k); //lower diag

                }

                c++;

            }
            // the first and last element on the secondary diagnaals which are not set on the loop
            U_X_CONV_Matrix(0,1) = rho*gridAnt->GetU(i,2,k);
            U_X_CONV_Matrix(c-1,c-2) = -rho*gridAnt->GetU(i,Nx+1-3,k);

            //border terms on            foont coonvective                        //the difusive
            U_X_Font(0)   +=  +rho*(gridSol->GetU(i,1,k))*       velocityField->GetU(i,2,k)            +sig*((gridSol->GetU(i,1,k)));
            U_X_Font(c-1) +=  -rho*(gridSol->GetU(i,Nx+1-2,k))  *velocityField->GetU(i,Nx+1-3,k)  +sig*(gridSol->GetU(i,Nx+1-2,k)); 


            U_X_Matrix = U_X_CONV_Matrix + U_X_DIFF_Matrix; //addint diffusive matrix to the convective


            U_X_SOL = U_X_Matrix.householderQr().solve(U_X_Font);


            c = 0;
            //substituting the line with the solution
            for(int j = 2;j<Nx+1-2;j++){
                gridSol->SetU(i,j,k,U_X_SOL(c));
                c++;
            }
    





        }
    }

}
void ADI::SolveADI_Y_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    
    for(int j = 2;j<Nx+1-2;j++){
        for(int k = 1;k<Nz-1;k++){
            int c=0;
            for(int i = 1;i<Ny-1;i++){

                double f_ijk = ADI::VelocityFontFunction(j*dh,i*dh + dh/2.0,k*dh + dh/2.0,time).u; //font function

                double X_DIFFUSION_TERM = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));

                double U_CONVECTIVE_TERM = velocityField->GetU(i,j,k)*((gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k)));
                double W_CONVECTIVE_TERM = velocityField->getWatU(i,j,k) *((gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1)));
                U_Y_Font(c) = gridAnt->GetU(i,j,k) + sig*(X_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != U_Y_CONV_Matrix.cols()-1 && c != 0){
                    U_Y_CONV_Matrix(c,c+1) =  rho*(velocityField->getVatU(i,j,k)); //upper diag
                    U_Y_CONV_Matrix(c,c-1) = -rho*(velocityField->getVatU(i,j,k)); //lower diag

                }

                c++;

            }

            U_Y_CONV_Matrix(0,1) = rho*(velocityField->getVatU(1,j,k));
            U_Y_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getVatU(Ny-1,j,k));

            //border terms on            foont coonvective                        //the difusive
            U_Y_Font(0)   +=  +rho*gridSol->GetU(0,j,k)* (velocityField->getVatU(1,j,k))   +sig*((gridSol->GetU(0,j,k)));
            U_Y_Font(c-1) +=  -rho*gridSol->GetU(Ny-1,j,k)*(velocityField->getVatU(Ny-2,j,k))   +sig*((gridSol->GetU(Ny-1,j,k)));

            U_Y_Matrix = U_Y_CONV_Matrix + U_Y_DIFF_Matrix; //addint diffusive matrix to the convective


            U_Y_SOL = U_Y_Matrix.householderQr().solve(U_Y_Font);


            c = 0;
            //substituting the line with the solution
            for(int i = 1;i<Ny-1;i++){
                gridSol->SetU(i,j,k,U_Y_SOL(c));
                c++;
            }
            //reset matrix after solving
            //U_Y_Matrix.diagonal() = VectorXd::Constant(U_Y_Matrix.cols(), 0);
            //for(int d=1;d<U_Y_Matrix.cols();d++){
            //    U_Y_Matrix(d,d-1) = 0.0;
            //    U_Y_Matrix(d-1,d) = 0.0;
            //}

        }
    }

        






}
void ADI::SolveADI_Z_U_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int i =1;i<Ny-1;i++){
        for(int j =2;j<Nx+1-2;j++){
            int c = 0;
            for(int k =1;k<Nz-1;k++){
                double f_ijk = ADI::VelocityFontFunction(j*dh,i*dh + dh/2.0,k*dh + dh/2.0,time).u; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                double X_DIFFUSION_TERM = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));

                double U_CONVECTIVE_TERM = (velocityField->GetU(i,j,k))*((gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k)));
                double V_CONVECTIVE_TERM = (velocityField->getVatU(i,j,k)) *((gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k)));

                U_Z_Font(c) = gridAnt->GetU(i,j,k) + sig*(Y_DIFFUSION_TERM+X_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+U_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;
                if(c != U_Z_CONV_Matrix.cols()-1 && c != 0){
                    U_Z_CONV_Matrix(c,c+1) =  rho*(velocityField->getWatU(i,j,k)); //upper diag
                    U_Z_CONV_Matrix(c,c-1) = -rho*(velocityField->getWatU(i,j,k)); //lower diag

                }
                c++;


            }

            U_Z_CONV_Matrix(0,1) = rho*(velocityField->getWatU(i,j,1));
            U_Z_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getWatU(i,j,Nz-1));

            U_Z_Font(0)   +=  +rho*gridSol->GetU(i,j,0)*velocityField->getWatU(i,j,1)   + sig*gridSol->GetU(i,j,0);
            U_Z_Font(c-1) +=  -rho*gridSol->GetU(i,j,Nz-1)*velocityField->getWatU(i,j,Nz-2)   +sig*gridSol->GetU(i,j,Nz-1);

            U_Z_Matrix = U_Z_CONV_Matrix + U_Z_DIFF_Matrix; //addint diffusive matrix to the convective

            U_Z_SOL = U_Z_Matrix.householderQr().solve(U_Z_Font);


            c = 0;
            //substituting the line with the solution
            for(int k = 1;k<Nz-1;k++){
                gridSol->SetU(i,j,k,U_Z_SOL(c));
                c++;
            }
            //reset matrix after solving
            //U_Z_Matrix.diagonal() = VectorXd::Constant(U_Z_Matrix.cols(), 0);
            //for(int d=1;d<U_Z_Matrix.cols();d++){
            //    U_Z_Matrix(d,d-1) = 0.0;
            //    U_Z_Matrix(d-1,d) = 0.0;
            //}


        }
    }
    


    
}

void ADI::SolveADI_X_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    for(int i = 2;i<Ny+1-2;i++){
        for(int k = 1;k<Nz-1;k++){
            int c = 0;
            for(int j = 1;j<Nx-1;j++){
                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh ,k*dh + dh/2.0,time).v; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));
                double V_CONVECTIVE_TERM = (velocityField->GetV(i,j,k)) *((gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k)));
                double W_CONVECTIVE_TERM = (velocityField->getWatV(i,j,k))*((gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1)));

                V_X_Font(c) = gridAnt->GetV(i,j,k) + sig*(Y_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != V_X_CONV_Matrix.cols()-1 && c != 0){
                    V_X_CONV_Matrix(c,c+1) =  rho*velocityField->getUatV(i,j,k); //upper diag
                    V_X_CONV_Matrix(c,c-1) = -rho*velocityField->getUatV(i,j,k); //lower diag

                }

                c++;

            }
            V_X_CONV_Matrix(0,1) = rho*gridAnt->getUatV(i,1,k);
            V_X_CONV_Matrix(c-1,c-2) = -rho*gridAnt->getUatV(i,Nx-1,k);

            V_X_Font(0)   +=  +rho*(gridSol->GetV(i,0,k))      *velocityField->getUatV(i,1,k)       +sig*((gridSol->GetV(i,0,k)));
            V_X_Font(c-1) +=  -rho*(gridSol->GetV(i,Nx-1,k))   *velocityField->getUatV(i,Nx-2,k)      +sig*(gridSol->GetV(i,Nx-1,k)); 

            V_X_Matrix = V_X_CONV_Matrix + V_X_DIFF_Matrix; //addint diffusive matrix to the convective


            V_X_SOL = V_X_Matrix.householderQr().solve(V_X_Font);



            c = 0;
            //substituting the line with the solution
            for(int j = 1;j<Nx-1;j++){
                gridSol->SetV(i,j,k,V_X_SOL(c));
                c++;
            }
            //reset matrix after solving
            //V_X_Matrix.diagonal() = VectorXd::Constant(V_X_Matrix.cols(), 0);
            //for(int d=1;d<V_X_Matrix.cols();d++){
            //    V_X_Matrix(d,d-1) = 0.0;
            //    V_X_Matrix(d-1,d) = 0.0;
            //}

        }

    }
}
void ADI::SolveADI_Y_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int j = 1;j<Nx-1;j++){
        for(int k = 1;k<Nz-1;k++){
            int c = 0;
            for(int i = 2;i<Ny+1-2;i++){
                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh,k*dh + dh/2.0,time).v; //font function

                double X_DIFFUSION_TERM = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));

                double U_CONVECTIVE_TERM = velocityField->getUatV(i,j,k)*((gridAnt->GetV(i,j+1,k) - gridAnt->GetV(i,j-1,k)));
                double W_CONVECTIVE_TERM = velocityField->getWatV(i,j,k) *((gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1)));

                V_Y_Font(c) = gridAnt->GetV(i,j,k) + sig*(X_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != V_Y_CONV_Matrix.cols()-1 && c != 0){
                    V_Y_CONV_Matrix(c,c+1) =  rho*(velocityField->GetV(i,j,k)); //upper diag
                    V_Y_CONV_Matrix(c,c-1) = -rho*(velocityField->GetV(i,j,k)); //lower diag

                }

                c++;

            }
          

            V_Y_CONV_Matrix(0,1)     =  rho*(velocityField->GetV(1,j,k));
            V_Y_CONV_Matrix(c-1,c-2) = -rho*(velocityField->GetV(Ny-1,j,k));

            V_Y_Font(0)   +=  +rho*gridSol->GetV(0,j,k)*   (velocityField->GetV(1,j,k))        +sig*((gridSol->GetV(0,j,k)));
            V_Y_Font(c-1) +=  -rho*gridSol->GetV(Ny-1,j,k)*(velocityField->GetV(Ny-2,j,k))     +sig*((gridSol->GetV(Ny-1,j,k)));

            V_Y_Matrix = V_Y_CONV_Matrix + V_Y_DIFF_Matrix; //addint diffusive matrix to the convective


            V_Y_SOL = V_Y_Matrix.householderQr().solve(V_Y_Font);
            


            c = 0;
            //substituting the line with the solution
            for(int i = 2;i<Ny+1-2;i++){
                gridSol->SetV(i,j,k,V_Y_SOL(c));
                c++;
            }
            
  
            


        }
    }


}
void ADI::SolveADI_Z_V_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int i = 2;i<Ny+1-2;i++){
        for(int j = 1;j<Nx-1;j++){
            int c = 0;
            for(int k = 1;k<Nz-1;k++){

                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh ,k*dh + dh/2.0,time).v; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                double X_DIFFUSION_TERM = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));

                double U_CONVECTIVE_TERM = (velocityField->getUatV(i,j,k)) *((gridAnt->GetV(i,j+1,k)  - gridAnt->GetV(i,j-1,k)));
                double V_CONVECTIVE_TERM = (velocityField->GetV(i,j,k)) *((gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k)));

                V_Z_Font(c) = gridAnt->GetV(i,j,k) + sig*(Y_DIFFUSION_TERM+X_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+U_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;
                if(c != V_Z_CONV_Matrix.cols()-1 && c != 0){
                    V_Z_CONV_Matrix(c,c+1) =  rho*(velocityField->getWatV(i,j,k)); //upper diag
                    V_Z_CONV_Matrix(c,c-1) = -rho*(velocityField->getWatV(i,j,k)); //lower diag

                }
                c++;


            }

            V_Z_CONV_Matrix(0,1) =      rho*(velocityField->getWatV(i,j,1));
            V_Z_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getWatV(i,j,Nz-1));

            V_Z_Font(0)   +=  +rho*gridSol->GetV(i,j,0)   *(velocityField->getWatV(i,j,1))       +sig*((gridSol->GetV(i,j,0)));
            V_Z_Font(c-1) +=  -rho*gridSol->GetV(i,j,Nz-1)*(velocityField->getWatV(i,j,Nz-2))    +sig*((gridSol->GetV(i,j,Nz-1)));

            V_Z_Matrix = V_Z_CONV_Matrix + V_Z_DIFF_Matrix; //addint diffusive matrix to the convective

            V_Z_SOL = V_Z_Matrix.householderQr().solve(V_Z_Font);

            c = 0;
            //substituting the line with the solution
            for(int k = 1;k<Nz-1;k++){
                gridSol->SetV(i,j,k,V_Z_SOL(c));
                c++;
            }
            //reset matrix after solving
            //V_Z_Matrix.diagonal() = VectorXd::Constant(V_Z_Matrix.cols(), 0);
            //for(int d=1;d<V_Z_Matrix.cols();d++){
            //    V_Z_Matrix(d,d-1) = 0.0;
            //    V_Z_Matrix(d-1,d) = 0.0;
            //}

        }
    }

}


void ADI::SolveADI_X_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    for(int i =1;i<Ny-1;i++){ 
        for(int k = 2;k<Nz+1-2;k++){ 
            int c = 0;
            for(int j = 1;j<Nx-1;j++){

                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh+ dh/2.0 ,k*dh ,time).w; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));
                double V_CONVECTIVE_TERM = (velocityField->getVatW(i,j,k)) *((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));
                double W_CONVECTIVE_TERM = (velocityField->GetW(i,j,k))*((gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1)));

                W_X_Font(c) = gridAnt->GetW(i,j,k) + sig*(Y_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != W_X_CONV_Matrix.cols()-1 && c != 0){
                    W_X_CONV_Matrix(c,c+1) =  rho*velocityField->getUatW(i,j,k); //upper diag
                    W_X_CONV_Matrix(c,c-1) = -rho*velocityField->getUatW(i,j,k); //lower diag

                }
                c++;

            }


            W_X_Matrix(0,1) = rho*gridAnt->getUatW(i,1,k);
            W_X_Matrix(c-1,c-2) = -rho*gridAnt->getUatW(i,Nx-1,k);

            W_X_Font(0)   +=  +rho*(gridSol->GetW(i,0,k))      *velocityField->getUatW(i,1,k)       +sig*((gridSol->GetW(i,0,k)));
            W_X_Font(c-1) +=  -rho*(gridSol->GetW(i,Nx-1,k))   *velocityField->getUatW(i,Nx-2,k)    +sig*(gridSol->GetW(i,Nx-1,k)); 

            W_X_Matrix = W_X_CONV_Matrix + W_X_DIFF_Matrix; //addint diffusive matrix to the convective


            W_X_SOL = W_X_Matrix.householderQr().solve(W_X_Font);
            c = 0;
            //substituting the line with the solution
            for(int j = 1;j<Nx-1;j++){
                gridSol->SetW(i,j,k,W_X_SOL(c));
                c++;
            }
            //reset matrix after solving
            //W_X_Matrix.diagonal() = VectorXd::Constant(W_X_Matrix.cols(), 0);
            //for(int d=1;d<W_X_Matrix.cols();d++){
            //    W_X_Matrix(d,d-1) = 0.0;
            //    W_X_Matrix(d-1,d) = 0.0;
            //}

        }
    }

}
void ADI::SolveADI_Y_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int j = 1;j<Nx-1;j++){
        for(int k = 2;k<Nz+1-2;k++){
            int c=0;
            for(int i = 1;i<Ny-1;i++){
                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh + dh/2.0,k*dh ,time).w; //font function

                double X_DIFFUSION_TERM = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));
                double Z_DIFFUSION_TERM = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));

                double U_CONVECTIVE_TERM = velocityField->getUatW(i,j,k)*((gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k)));
                double W_CONVECTIVE_TERM = velocityField->GetW(i,j,k) *((gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1)));

                W_Y_Font(c) = gridAnt->GetW(i,j,k) + sig*(X_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM+W_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;

                if(c != W_Y_CONV_Matrix.cols()-1 && c != 0){
                    W_Y_CONV_Matrix(c,c+1) =  rho*(velocityField->getVatW(i,j,k)); //upper diag
                    W_Y_CONV_Matrix(c,c-1) = -rho*(velocityField->getVatW(i,j,k)); //lower diag

                }

                c++;
            }

            W_Y_CONV_Matrix(0,1)     =  rho*(velocityField->getVatW(1,j,k));
            W_Y_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getVatW(Ny-1,j,k));

            W_Y_Font(0)   +=  +rho*gridSol->GetW(0,j,k)*   (velocityField->getVatW(1,j,k))        +sig*((gridSol->GetW(0,j,k)));
            W_Y_Font(c-1) +=  -rho*gridSol->GetW(Ny-1,j,k)*(velocityField->getVatW(Ny-2,j,k))     +sig*((gridSol->GetW(Ny-1,j,k)));

            W_Y_Matrix = W_Y_CONV_Matrix + W_Y_DIFF_Matrix; //addint diffusive matrix to the convective


            W_Y_SOL = W_Y_Matrix.householderQr().solve(W_Y_Font);


            c = 0;
            //substituting the line with the solution
            for(int i = 1;i<Ny-1;i++){
                gridSol->SetW(i,j,k,W_Y_SOL(c));
                c++;
            }




        }
    }

}
void ADI::SolveADI_Z_W_Step(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time){
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    for(int i =1;i<Ny-1;i++){
        for(int j =1;j<Nx-1;j++){
            int c = 0;
            for(int k =2;k<Nz+1-2;k++){
                double f_ijk = ADI::VelocityFontFunction(j*dh+ dh/2.0,i*dh + dh/2.0,k*dh ,time).w; //font function

                double Y_DIFFUSION_TERM = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                double X_DIFFUSION_TERM = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));

                double U_CONVECTIVE_TERM = (velocityField->getUatW(i,j,k))*((gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k)));
                double V_CONVECTIVE_TERM = (velocityField->getVatW(i,j,k)) *((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));

                W_Z_Font(c) = gridAnt->GetW(i,j,k) + sig*(Y_DIFFUSION_TERM+X_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+U_CONVECTIVE_TERM)  - (dt/3.0)*f_ijk;
                if(c != W_Z_CONV_Matrix.cols()-1 && c != 0){
                    W_Z_CONV_Matrix(c,c+1) =  rho*(velocityField->GetW(i,j,k)); //upper diag
                    W_Z_CONV_Matrix(c,c-1) = -rho*(velocityField->GetW(i,j,k)); //lower diag

                }
                c++;

            }


            W_Z_CONV_Matrix(0,1) = rho*(velocityField->GetW(i,j,2));
            W_Z_CONV_Matrix(c-1,c-2) = -rho*(velocityField->GetW(i,j,Nz+1-3));

            W_Z_Font(0)   +=  +rho*gridSol->GetW(i,j,1)*(velocityField->GetW(i,j,2))   +sig*((gridSol->GetW(i,j,1)));
            W_Z_Font(c-1) +=  -rho*gridSol->GetW(i,j,Nz+1-2)*(velocityField->GetW(i,j,Nz+1-3))   +sig*((gridSol->GetW(i,j,Nz+1-2)));

            W_Z_Matrix = W_Z_CONV_Matrix + W_Z_DIFF_Matrix; //addint diffusive matrix to the convective

            W_Z_SOL = W_Z_Matrix.householderQr().solve(W_Z_Font);


            c = 0;
            //substituting the line with the solution
            for(int k = 2;k<Nz+1-2;k++){
                gridSol->SetW(i,j,k,W_Z_SOL(c));
                c++;
            }

        }
    }



}

*/

//PARALLEL OPENMP TESTING


void ADI::SolveADI_X_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));


    
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
            for(int k = 1; k < Nz-1; k++) {
                int c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh, i*dh + dh/2.0, k*dh + dh/2.0, time).u;
                    //ADI::VelocityBorderFunction(j*dh, i*dh + dh/2.0, k*dh + dh/2.0, time).u;

                    double Y_DIFFUSION_TERM = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));
                    double V_CONVECTIVE_TERM = velocityField->getVatU(i,j,k)*((gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k)));
                    double W_CONVECTIVE_TERM = velocityField->getWatU(i,j,k)*((gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1)));

                    local_U_X_Font(c) = gridAnt->GetU(i,j,k) + sig*(Y_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_U_X_CONV_Matrix.cols()-1 && c != 0) {
                        local_U_X_CONV_Matrix(c,c+1) = rho*velocityField->GetU(i,j,k);
                        local_U_X_CONV_Matrix(c,c-1) = -rho*velocityField->GetU(i,j,k);
                    }
                    c++;
                }

                local_U_X_CONV_Matrix(0,1) =      rho*gridAnt->GetU(i,2,k);
                local_U_X_CONV_Matrix(c-1,c-2) = -rho*gridAnt->GetU(i,Nx+1-3,k);

                local_U_X_Font(0)   +=  rho*gridSol->GetU(i,1,k)     *velocityField->GetU(i,2,k) +      sig*((gridSol->GetU(i,1,k)));
                local_U_X_Font(c-1) += -rho*gridSol->GetU(i,Nx+1-2,k)*velocityField->GetU(i,Nx+1-3,k) + sig*(gridSol->GetU(i,Nx+1-2,k));

                local_U_X_Matrix = local_U_X_CONV_Matrix + local_U_X_DIFF_Matrix;

                //SOLID MASK CHECKING
                //yes, it is inefficient, but honestly, we are already solving so fast it barely makes a difference
                //and this is the easiest way to implement this
                
                c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j-1,k) == SOLID_CELL ){
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
                    gridSol->SetU(i,j,k, local_U_X_SOL(c));
                    c++;
                }

                
            }
        }
    }
}
void ADI::SolveADI_Y_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    
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
            for(int k = 1; k < Nz-1; k++) {
                int c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh, i*dh + dh/2.0, k*dh + dh/2.0, time).u;
                    

                    double X_DIFFUSION_TERM = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));

                    double U_CONVECTIVE_TERM = velocityField->GetU(i,j,k)*((gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k)));
                    double W_CONVECTIVE_TERM = velocityField->getWatU(i,j,k)*((gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1)));
                    
                    local_U_Y_Font(c) = gridAnt->GetU(i,j,k) + sig*(X_DIFFUSION_TERM+Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM+W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_U_Y_CONV_Matrix.cols()-1 && c != 0) {
                        local_U_Y_CONV_Matrix(c,c+1) = rho*(velocityField->getVatU(i,j,k));
                        local_U_Y_CONV_Matrix(c,c-1) = -rho*(velocityField->getVatU(i,j,k));
                    }
                    c++;
                }

                local_U_Y_CONV_Matrix(0,1) =      rho*(velocityField->getVatU(1,j,k));
                local_U_Y_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getVatU(Ny-2,j,k));

                // Border terms
                local_U_Y_Font(0)   +=   rho*velocityField->getVatU(1,j,k)*gridSol->GetU(0,j,k)       + sig*((gridSol->GetU(0,j,k)));
                local_U_Y_Font(c-1) +=  -rho*velocityField->getVatU(Ny-2,j,k)*gridSol->GetU(Ny-1,j,k) + sig*((gridSol->GetU(Ny-1,j,k)));

                local_U_Y_Matrix = local_U_Y_CONV_Matrix + local_U_Y_DIFF_Matrix;

                c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j-1,k) == SOLID_CELL){
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
                    gridSol->SetU(i,j,k, local_U_Y_SOL(c));
                    c++;
                }
            }
        }
    }
}
void ADI::SolveADI_Z_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_U_Z_CONV_Matrix = U_Z_CONV_Matrix;
        Eigen::MatrixXd local_U_Z_DIFF_Matrix = U_Z_DIFF_Matrix;
        Eigen::VectorXd local_U_Z_Font = U_Z_Font;
        Eigen::VectorXd local_U_Z_SOL;
        Eigen::MatrixXd local_U_Z_Matrix;

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int j = 2; j < Nx+1-2; j++) {
                int c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh, i*dh + dh/2.0, k*dh + dh/2.0, time).u;

                    double Y_DIFFUSION_TERM = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                    double X_DIFFUSION_TERM = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));

                    double U_CONVECTIVE_TERM = (velocityField->GetU(i,j,k))*((gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k)));
                    double V_CONVECTIVE_TERM = (velocityField->getVatU(i,j,k)) *((gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k)));

                    local_U_Z_Font(c) = gridAnt->GetU(i,j,k) + sig*(Y_DIFFUSION_TERM+X_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM+U_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_U_Z_CONV_Matrix.cols()-1 && c != 0) {
                        local_U_Z_CONV_Matrix(c,c+1) =  rho*(velocityField->getWatU(i,j,k));
                        local_U_Z_CONV_Matrix(c,c-1) = -rho*(velocityField->getWatU(i,j,k));
                    }
                    c++;
                }

                // Set boundary elements
                local_U_Z_CONV_Matrix(0,1) = rho*(velocityField->getWatU(i,j,1));
                local_U_Z_CONV_Matrix(c-1,c-2) = -rho*(velocityField->getWatU(i,j,Nz-2));

                // Apply boundary conditions
                local_U_Z_Font(0) +=   rho*gridSol->GetU(i,j,0)*velocityField->getWatU(i,j,1) + sig*gridSol->GetU(i,j,0);
                local_U_Z_Font(c-1) += -rho*gridSol->GetU(i,j,Nz-1)*velocityField->getWatU(i,j,Nz-2) + sig*gridSol->GetU(i,j,Nz-1);

                // Solve the system
                local_U_Z_Matrix = local_U_Z_CONV_Matrix + local_U_Z_DIFF_Matrix;

                c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j-1,k) == SOLID_CELL ){
                        local_U_Z_Font(c) = 0.0;
                        local_U_Z_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_U_Z_Matrix(c,c-1) = 0.0; 
                            local_U_Z_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_U_Z_CONV_Matrix.cols()-1){
                            local_U_Z_Matrix(c,c+1) = 0.0;
                            local_U_Z_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }
                TDMA(local_U_Z_Matrix,local_U_Z_Font,local_U_Z_SOL);
                // Store the solution
                c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    gridSol->SetU(i,j,k, local_U_Z_SOL(c));
                    c++;
                }
            }
        }
    }
}

void ADI::SolveADI_X_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));

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
            for(int k = 1; k < Nz-1; k++) {
                int c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh, k*dh + dh/2.0, time).v;

                    double Y_DIFFUSION_TERM = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));
                    double V_CONVECTIVE_TERM = (velocityField->GetV(i,j,k)) * ((gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k)));
                    double W_CONVECTIVE_TERM = (velocityField->getWatV(i,j,k)) * ((gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1)));

                    local_V_X_Font(c) = gridAnt->GetV(i,j,k) + sig*(Y_DIFFUSION_TERM + Z_DIFFUSION_TERM) - rho*(V_CONVECTIVE_TERM + W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_V_X_CONV_Matrix.cols()-1 && c != 0) {
                        local_V_X_CONV_Matrix(c,c+1) = rho * velocityField->getUatV(i,j,k);
                        local_V_X_CONV_Matrix(c,c-1) = -rho * velocityField->getUatV(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_V_X_CONV_Matrix(0,1) =      rho * velocityField->getUatV(i,1,k);
                local_V_X_CONV_Matrix(c-1,c-2) = -rho * velocityField->getUatV(i,Nx-2,k);

                // Apply boundary conditions
                local_V_X_Font(0)   +=  rho  * gridSol->GetV(i,0,k)    * velocityField->getUatV(i,1,k)    + sig * (gridSol->GetV(i,0,k));
                local_V_X_Font(c-1) += -rho * gridSol->GetV(i,Nx-1,k) * velocityField->getUatV(i,Nx-2,k) + sig * (gridSol->GetV(i,Nx-1,k));

                // Solve the system
                local_V_X_Matrix = local_V_X_CONV_Matrix + local_V_X_DIFF_Matrix;

                c = 0;
                for(int j = 1; j < Nx-1; j++)  {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i-1,j,k) == SOLID_CELL ){
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
                    gridSol->SetV(i,j,k, local_V_X_SOL(c));
                    c++;
                }
            }
        }
    }
}
void ADI::SolveADI_Y_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));

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
            for(int k = 1; k < Nz-1; k++) {
                int c = 0;
                for(int i = 2; i < Ny+1-2; i++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh, k*dh + dh/2.0, time).v;

                    double X_DIFFUSION_TERM = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));

                    double U_CONVECTIVE_TERM = velocityField->getUatV(i,j,k) * ((gridAnt->GetV(i,j+1,k) - gridAnt->GetV(i,j-1,k)));
                    double W_CONVECTIVE_TERM = velocityField->getWatV(i,j,k) * ((gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1)));

                    local_V_Y_Font(c) = gridAnt->GetV(i,j,k) + sig*(X_DIFFUSION_TERM + Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM + W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_V_Y_CONV_Matrix.cols()-1 && c != 0) {
                        //TAYLOR_GREEN_VORTEX_VELOCITY(j*dh + dh/2.0, i*dh, k*dh + dh/2.0, time).v;
                        local_V_Y_CONV_Matrix(c,c+1) =  rho * velocityField->GetV(i,j,k);
                        local_V_Y_CONV_Matrix(c,c-1) = -rho * velocityField->GetV(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_V_Y_CONV_Matrix(0,1) = rho * velocityField->GetV(2,j,k);
                local_V_Y_CONV_Matrix(c-1,c-2) = -rho * velocityField->GetV(Ny+1-3,j,k);
                // Apply boundary conditions
                local_V_Y_Font(0)   +=   rho * gridSol->GetV(1,j,k)    * velocityField->GetV(2,j,k)    + sig * gridSol->GetV(1,j,k);
                local_V_Y_Font(c-1) +=  -rho * gridSol->GetV(Ny+1-2,j,k) * velocityField->GetV(Ny+1-3,j,k) + sig * gridSol->GetV(Ny+1-2,j,k);

                //if(j == 4 && k == 4){
                //    exportVectorToFile(local_V_Y_Font,"Exports/EigenVector/V_Y_FONT.csv");
                //    exportMatrixToFile(local_V_Y_Matrix,"Exports/EigenVector/V_Y_MATRIX.csv");
                //}

                // Solve the system
                local_V_Y_Matrix = local_V_Y_CONV_Matrix + local_V_Y_DIFF_Matrix;
                c = 0;
                for(int i = 2; i < Ny+1-2; i++)  {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i-1,j,k) == SOLID_CELL){
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
                    gridSol->SetV(i,j,k, local_V_Y_SOL(c));
                    c++;
                }
            }
        }
    }
}
void ADI::SolveADI_Z_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_V_Z_CONV_Matrix = V_Z_CONV_Matrix;
        Eigen::MatrixXd local_V_Z_DIFF_Matrix = V_Z_DIFF_Matrix;
        Eigen::VectorXd local_V_Z_Font = V_Z_Font;
        Eigen::VectorXd local_V_Z_SOL;
        Eigen::MatrixXd local_V_Z_Matrix;

        #pragma omp for
        for(int i = 2; i < Ny+1-2; i++) {
            for(int j = 1; j < Nx-1; j++) {
                int c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh, k*dh + dh/2.0, time).v;

                    double Y_DIFFUSION_TERM = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                    double X_DIFFUSION_TERM = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));

                    double U_CONVECTIVE_TERM = (velocityField->getUatV(i,j,k)) * ((gridAnt->GetV(i,j+1,k) - gridAnt->GetV(i,j-1,k)));
                    double V_CONVECTIVE_TERM = (velocityField->GetV(i,j,k)) * ((gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k)));

                    local_V_Z_Font(c) = gridAnt->GetV(i,j,k) + sig*(Y_DIFFUSION_TERM + X_DIFFUSION_TERM)  - rho*(V_CONVECTIVE_TERM + U_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_V_Z_CONV_Matrix.cols()-1 && c != 0) {
                        local_V_Z_CONV_Matrix(c,c+1) = rho * velocityField->getWatV(i,j,k);
                        local_V_Z_CONV_Matrix(c,c-1) = -rho * velocityField->getWatV(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_V_Z_CONV_Matrix(0,1) = rho * velocityField->getWatV(i,j,1);
                local_V_Z_CONV_Matrix(c-1,c-2) = -rho * velocityField->getWatV(i,j,Nz-2);

                // Apply boundary conditions
                local_V_Z_Font(0)   +=  rho * gridSol->GetV(i,j,0)     * velocityField->getWatV(i,j,1)    + sig * gridSol->GetV(i,j,0);
                local_V_Z_Font(c-1) += -rho * gridSol->GetV(i,j,Nz-1)  * velocityField->getWatV(i,j,Nz-2) + sig * gridSol->GetV(i,j,Nz-1);

                // Solve the system
                local_V_Z_Matrix = local_V_Z_CONV_Matrix + local_V_Z_DIFF_Matrix;

                c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i-1,j,k) == SOLID_CELL){
                        local_V_Z_Font(c) = 0.0;
                        local_V_Z_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_V_Z_Matrix(c,c-1) = 0.0; 
                            local_V_Z_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_V_Z_CONV_Matrix.cols()-1){
                            local_V_Z_Matrix(c,c+1) = 0.0;
                            local_V_Z_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }

                TDMA(local_V_Z_Matrix,local_V_Z_Font,local_V_Z_SOL);
                // Store the solution
                c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    gridSol->SetV(i,j,k, local_V_Z_SOL(c));
                    c++;
                }
            }
        }
    }
}

void ADI::SolveADI_X_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_W_X_CONV_Matrix = W_X_CONV_Matrix;
        Eigen::MatrixXd local_W_X_DIFF_Matrix = W_X_DIFF_Matrix;
        Eigen::VectorXd local_W_X_Font = W_X_Font;
        Eigen::VectorXd local_W_X_SOL;
        Eigen::MatrixXd local_W_X_Matrix;

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int k = 2; k < Nz+1-2; k++) {
                int c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh + dh/2.0, k*dh, time).w;

                    double Y_DIFFUSION_TERM = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));
                    double V_CONVECTIVE_TERM = (velocityField->getVatW(i,j,k)) * ((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));
                    double W_CONVECTIVE_TERM = (velocityField->GetW(i,j,k)) * ((gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1)));

                    local_W_X_Font(c) = gridAnt->GetW(i,j,k) + sig*(Y_DIFFUSION_TERM + Z_DIFFUSION_TERM)  - rho*(V_CONVECTIVE_TERM + W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_W_X_CONV_Matrix.cols()-1 && c != 0) {
                        local_W_X_CONV_Matrix(c,c+1) =  rho * velocityField->getUatW(i,j,k);
                        local_W_X_CONV_Matrix(c,c-1) = -rho * velocityField->getUatW(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_W_X_CONV_Matrix(0,1) =      rho * velocityField->getUatW(i,1,k);
                local_W_X_CONV_Matrix(c-1,c-2) = -rho * velocityField->getUatW(i,Nx-2,k);

                // Apply boundary conditions
                local_W_X_Font(0) += rho * gridSol->GetW(i,0,k) * velocityField->getUatW(i,1,k) + sig * gridSol->GetW(i,0,k);
                local_W_X_Font(c-1) += -rho * gridSol->GetW(i,Nx-1,k) * velocityField->getUatW(i,Nx-2,k) + sig * gridSol->GetW(i,Nx-1,k);

                // Solve the system
                local_W_X_Matrix = local_W_X_CONV_Matrix + local_W_X_DIFF_Matrix;
                c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j,k-1) == SOLID_CELL){
                        local_W_X_Font(c) = 0.0;
                        local_W_X_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_W_X_Matrix(c,c-1) = 0.0; 
                            local_W_X_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_W_X_CONV_Matrix.cols()-1){
                            local_W_X_Matrix(c,c+1) = 0.0;
                            local_W_X_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }
                TDMA(local_W_X_Matrix,local_W_X_Font,local_W_X_SOL);
                // Store the solution
                c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    gridSol->SetW(i,j,k, local_W_X_SOL(c));
                    c++;
                }
            }
        }
    }
}
void ADI::SolveADI_Y_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_W_Y_CONV_Matrix = W_Y_CONV_Matrix;
        Eigen::MatrixXd local_W_Y_DIFF_Matrix = W_Y_DIFF_Matrix;
        Eigen::VectorXd local_W_Y_Font = W_Y_Font;
        Eigen::VectorXd local_W_Y_SOL;
        Eigen::MatrixXd local_W_Y_Matrix;

        #pragma omp for
        for(int j = 1; j < Nx-1; j++) {
            for(int k = 2; k < Nz+1-2; k++) {
                int c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh + dh/2.0, k*dh, time).w;

                    double X_DIFFUSION_TERM = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));
                    double Z_DIFFUSION_TERM = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));

                    double U_CONVECTIVE_TERM = velocityField->getUatW(i,j,k) * ((gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k)));
                    double W_CONVECTIVE_TERM = velocityField->GetW(i,j,k) * ((gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1)));

                    local_W_Y_Font(c) = gridAnt->GetW(i,j,k) + sig*(X_DIFFUSION_TERM + Z_DIFFUSION_TERM) - rho*(U_CONVECTIVE_TERM + W_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_W_Y_CONV_Matrix.cols()-1 && c != 0) {
                        local_W_Y_CONV_Matrix(c,c+1) = rho * velocityField->getVatW(i,j,k);
                        local_W_Y_CONV_Matrix(c,c-1) = -rho * velocityField->getVatW(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_W_Y_CONV_Matrix(0,1) =      rho * velocityField->getVatW(1,j,k);
                local_W_Y_CONV_Matrix(c-1,c-2) = -rho * velocityField->getVatW(Ny-2,j,k);

                // Apply boundary conditions
                local_W_Y_Font(0) += rho * gridSol->GetW(0,j,k) * velocityField->getVatW(1,j,k) + sig * gridSol->GetW(0,j,k);
                local_W_Y_Font(c-1) += -rho * gridSol->GetW(Ny-1,j,k) * velocityField->getVatW(Ny-2,j,k) + sig * gridSol->GetW(Ny-1,j,k);

                // Solve the system
                local_W_Y_Matrix = local_W_Y_CONV_Matrix + local_W_Y_DIFF_Matrix;
                c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j,k-1) == SOLID_CELL){
                        local_W_Y_Font(c) = 0.0;
                        local_W_Y_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_W_Y_Matrix(c,c-1) = 0.0; 
                            local_W_Y_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_W_Y_CONV_Matrix.cols()-1){
                            local_W_Y_Matrix(c,c+1) = 0.0;
                            local_W_Y_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }
                TDMA(local_W_Y_Matrix,local_W_Y_Font,local_W_Y_SOL);
                // Store the solution
                c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    gridSol->SetW(i,j,k, local_W_Y_SOL(c));
                    c++;
                }
            }
        }
    }
}
void ADI::SolveADI_Z_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI::dh;
    double dt = ADI::dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));

    #pragma omp parallel
    {
        // Create thread-local copies of matrices/vectors
        Eigen::MatrixXd local_W_Z_CONV_Matrix = W_Z_CONV_Matrix;
        Eigen::MatrixXd local_W_Z_DIFF_Matrix = W_Z_DIFF_Matrix;
        Eigen::VectorXd local_W_Z_Font = W_Z_Font;
        Eigen::VectorXd local_W_Z_SOL;
        Eigen::MatrixXd local_W_Z_Matrix;

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int j = 1; j < Nx-1; j++) {
                int c = 0;
                for(int k = 2; k < Nz+1-2; k++) {
                    double f_ijk = ADI::VelocityFontFunction(j*dh + dh/2.0, i*dh + dh/2.0, k*dh, time).w;

                    double Y_DIFFUSION_TERM = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                    double X_DIFFUSION_TERM = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));

                    double U_CONVECTIVE_TERM = (velocityField->getUatW(i,j,k)) * ((gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k)));
                    double V_CONVECTIVE_TERM = (velocityField->getVatW(i,j,k)) * ((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));

                    local_W_Z_Font(c) = gridAnt->GetW(i,j,k) + sig*(Y_DIFFUSION_TERM + X_DIFFUSION_TERM)  - rho*(V_CONVECTIVE_TERM + U_CONVECTIVE_TERM) - (dt/3.0)*f_ijk;

                    if(c != local_W_Z_CONV_Matrix.cols()-1 && c != 0) {
                        local_W_Z_CONV_Matrix(c,c+1) = rho * velocityField->GetW(i,j,k);
                        local_W_Z_CONV_Matrix(c,c-1) = -rho * velocityField->GetW(i,j,k);
                    }
                    c++;
                }

                // Set boundary elements
                local_W_Z_CONV_Matrix(0,1) = rho * velocityField->GetW(i,j,2);
                local_W_Z_CONV_Matrix(c-1,c-2) = -rho * velocityField->GetW(i,j,Nz+1-3);

                // Apply boundary conditions
                local_W_Z_Font(0) += rho * gridSol->GetW(i,j,1) * velocityField->GetW(i,j,2) + sig * gridSol->GetW(i,j,1);
                local_W_Z_Font(c-1) += -rho * gridSol->GetW(i,j,Nz+1-2) * velocityField->GetW(i,j,Nz+1-3) + sig * gridSol->GetW(i,j,Nz+1-2);

                // Solve the system
                local_W_Z_Matrix = local_W_Z_CONV_Matrix + local_W_Z_DIFF_Matrix;
                c = 0;
                for(int k = 2; k < Nz+1-2; k++) {
                    if(gridSol->GetSolid(i,j,k) == SOLID_CELL || gridSol->GetSolid(i,j,k-1) == SOLID_CELL){
                        local_W_Z_Font(c) = 0.0;
                        local_W_Z_Matrix(c,c) = 1.0;
                        if(c >=1){
                            local_W_Z_Matrix(c,c-1) = 0.0; 
                            local_W_Z_Matrix(c-1,c) = 0.0; 
                        }
                        if(c < local_W_Z_CONV_Matrix.cols()-1){
                            local_W_Z_Matrix(c,c+1) = 0.0;
                            local_W_Z_Matrix(c+1,c) = 0.0;
                        }
                    }
                    
                    c++;
                }
                TDMA(local_W_Z_Matrix,local_W_Z_Font,local_W_Z_SOL);
                // Store the solution
                c = 0;
                for(int k = 2; k < Nz+1-2; k++) {
                    gridSol->SetW(i,j,k, local_W_Z_SOL(c));
                    c++;
                }
            }
        }
    }
}



void ADI::InitializeADI(MAC* grid,double dt,Vec3(*VelocityBorderFunction)(double, double, double,double),Vec3(*VelocityFont)(double, double, double,double),double(*PressureFunction)(double, double, double,double)){
    int Nx = grid->Nx;
    int Ny = grid->Ny;
    int Nz = grid->Nz;

    ADI::VelocityBorderFunction = VelocityBorderFunction;
    ADI::VelocityFontFunction = VelocityFont;
    ADI::PressureFunction = PressureFunction;


    ADI::dt = dt;
    ADI::dh = grid->dh;
    
    //sigma coefficient
    ADI::SIG = (dt/(dh*dh*3.0))* (SIMULATION.EPS);


    
    //please pay lots of attention here, since we have different ddirection and nodes, we have lots of matrixes and fonts, we pre allocated them to save time later
    //X DIR MATRIXES
    int u_size = (Nx + 1) - 4;   // U matrix size at x dir has less nodes, check diagram
    int v_size = (Nx)     - 2;   // V matrix size is normal
    int w_size = (Nx)     - 2;   // W matrix size
    
    ADI::U_X_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_X_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_X_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_X_CONV_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_X_CONV_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_X_CONV_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_X_DIFF_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_X_DIFF_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_X_DIFF_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_X_Font = VectorXd::Zero(u_size);
    ADI::V_X_Font = VectorXd::Zero(v_size);
    ADI::W_X_Font = VectorXd::Zero(w_size);

    ADI::U_X_SOL = VectorXd::Zero(u_size);
    ADI::V_X_SOL = VectorXd::Zero(v_size);
    ADI::W_X_SOL = VectorXd::Zero(w_size);
    
    // Set main diagonals
    //ADI::SIG = (dt / (grid->dh * 3.0)); past definition
    U_X_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, (2 * SIG) + 1);
    V_X_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, (2 * SIG)+ 1);
    W_X_DIFF_Matrix.diagonal() = VectorXd::Constant(w_size, (2 * SIG )+ 1);


    
    // Set sub-diagonals (-SIG values), will also work for a 4x4 in one value bcause for is an if too
    for (int i = 1; i < u_size; i++) {
        U_X_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
        U_X_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
    }
    
    for (int i = 1; i < v_size; i++) {
        V_X_DIFF_Matrix(i, i-1) = -SIG;      
        V_X_DIFF_Matrix(i-1, i) = -SIG;    
    }
    
    for (int i = 1; i < w_size; i++) {
        W_X_DIFF_Matrix(i, i-1) = -SIG;     
        W_X_DIFF_Matrix(i-1, i) = -SIG;      
    }




    // Y DIR MATRIXES
    u_size = (Ny )    - 2;   
    v_size = (Ny+1)   - 4;   
    w_size = (Ny)     - 2;   

    ADI::U_Y_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Y_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Y_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_Y_CONV_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Y_CONV_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Y_CONV_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_Y_DIFF_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Y_DIFF_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Y_DIFF_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_Y_Font=  VectorXd::Zero(u_size);
    ADI::V_Y_Font = VectorXd::Zero(v_size);
    ADI::W_Y_Font = VectorXd::Zero(w_size);

    ADI::U_Y_SOL=  VectorXd::Zero(u_size);
    ADI::V_Y_SOL = VectorXd::Zero(v_size);
    ADI::W_Y_SOL = VectorXd::Zero(w_size);
    
    // Set main diagonals
    U_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, 2 * SIG + 1);
    V_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, 2 * SIG + 1);
    W_Y_DIFF_Matrix.diagonal() = VectorXd::Constant(w_size, 2 * SIG + 1);


    for (int i = 1; i < u_size; i++) {
        U_Y_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
        U_Y_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
    }
    
    for (int i = 1; i < v_size; i++) {
        V_Y_DIFF_Matrix(i, i-1) = -SIG;      
        V_Y_DIFF_Matrix(i-1, i) = -SIG;    
    }
    
    for (int i = 1; i < w_size; i++) {
        W_Y_DIFF_Matrix(i, i-1) = -SIG;     
        W_Y_DIFF_Matrix(i-1, i) = -SIG;      
    }

    //z matrixes
    u_size = (Nz )    - 2;   
    v_size = (Nz)     - 2;   
    w_size = (Nz+1)   - 4;   

    ADI::U_Z_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Z_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Z_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_Z_CONV_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Z_CONV_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Z_CONV_Matrix = MatrixXd::Zero(w_size, w_size);


    ADI::U_Z_DIFF_Matrix = MatrixXd::Zero(u_size, u_size);
    ADI::V_Z_DIFF_Matrix = MatrixXd::Zero(v_size, v_size);
    ADI::W_Z_DIFF_Matrix = MatrixXd::Zero(w_size, w_size);

    ADI::U_Z_Font=  VectorXd::Zero(u_size);
    ADI::V_Z_Font = VectorXd::Zero(v_size);
    ADI::W_Z_Font = VectorXd::Zero(w_size);

    ADI::U_Z_SOL  = VectorXd::Zero(u_size);
    ADI::V_Z_SOL  = VectorXd::Zero(v_size);
    ADI::W_Z_SOL  = VectorXd::Zero(w_size);
    
    // Set main diagonals
    U_Z_DIFF_Matrix.diagonal() = VectorXd::Constant(u_size, 2 * SIG + 1);
    V_Z_DIFF_Matrix.diagonal() = VectorXd::Constant(v_size, 2 * SIG + 1);
    W_Z_DIFF_Matrix.diagonal() = VectorXd::Constant(w_size, 2 * SIG + 1);

    for (int i = 1; i < u_size; i++) {
        U_Z_DIFF_Matrix(i, i-1) = -SIG;      // Lower diagonal
        U_Z_DIFF_Matrix(i-1, i) = -SIG;      // Upper diagonal
    }
    
    for (int i = 1; i < v_size; i++) {
        V_Z_DIFF_Matrix(i, i-1) = -SIG;      
        V_Z_DIFF_Matrix(i-1, i) = -SIG;    
    }
    
    for (int i = 1; i < w_size; i++) {
        W_Z_DIFF_Matrix(i, i-1) = -SIG;     
        W_Z_DIFF_Matrix(i-1, i) = -SIG;      
    }

    ADI::X_STEP_SOL.InitializeGrid(grid->omega);
    ADI::Y_STEP_SOL.InitializeGrid(grid->omega);
    ADI::Z_STEP_SOL.InitializeGrid(grid->omega);

    ADI::X_STEP_SOL.CopyGrid(*grid);
    ADI::Y_STEP_SOL.CopyGrid(*grid);
    ADI::Z_STEP_SOL.CopyGrid(*grid);



}

//IM BREAKING the component on the exit
void ADI::SolveADIStep(MAC* gridAnt,MAC* gridSol,double time){




    double start = omp_get_wtime();


 
    time += ADI::dt/3.0;
    ADI::X_STEP_SOL.SetBorder(ADI::VelocityBorderFunction,ADI::PressureFunction,time);
    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) ADI::X_STEP_SOL.SetNeumannBorder();
    SolveADI_X_U_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);
    SolveADI_X_V_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);
    SolveADI_X_W_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);


    time += ADI::dt/3.0;
    ADI::Y_STEP_SOL.SetBorder(ADI::VelocityBorderFunction,ADI::PressureFunction,time);
    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) ADI::Y_STEP_SOL.SetNeumannBorder();
    SolveADI_Y_U_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);
    SolveADI_Y_V_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);
    SolveADI_Y_W_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);


    time += ADI::dt/3.0;
    ADI::Z_STEP_SOL.SetBorder(ADI::VelocityBorderFunction,ADI::PressureFunction,time);
    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE) ADI::Z_STEP_SOL.SetNeumannBorder();
    SolveADI_Z_U_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);
    SolveADI_Z_V_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);
    SolveADI_Z_W_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);

    double end = omp_get_wtime();

    SIMULATION.lastADISolveTime = end - start;

    //WriteToCSV("Data/Times/time_adi_16.csv",end-start);

    //dont know if i need this!


    gridSol->CopyGrid(Z_STEP_SOL);


}