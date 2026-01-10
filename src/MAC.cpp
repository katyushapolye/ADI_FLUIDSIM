#include "headers/Core/MAC.h"


#define FOR_EACH_2D_NON_BOUNDARY_U_FACE(code_block) \
  for (int i = 1; i < SIMULATION.Ny-1; i++) { \
    for (int j = 2; j < SIMULATION.Nx+1-2; j++) { \
      code_block \
    } \
  }

  #define FOR_EACH_2D_NON_BOUNDARY_V_FACE(code_block) \
  for (int i = 2; i < SIMULATION.Ny+1-2; i++) { \
    for (int j = 1; j < SIMULATION.Nx-1; j++) { \
      code_block \
    } \
  }

  #define FOR_EACH_2D_NON_BOUNDARY_CELL(code_block) \
  for (int i = 1; i < SIMULATION.Ny-1; i++) { \
    for (int j = 1; j < SIMULATION.Nx-1; j++) { \
      code_block \
    } \
  }

// 3D Macros for looping over non-boundary faces and cells
#define FOR_EACH_3D_NON_BOUNDARY_U_FACE(code_block) \
  for (int i = 1; i < SIMULATION.Ny-1; i++) { \
    for (int j = 2; j < SIMULATION.Nx+1-2; j++) { \
      for (int k = 1; k < SIMULATION.Nz-1; k++) { \
        code_block \
      } \
    } \
  }


#define FOR_EACH_3D_NON_BOUNDARY_V_FACE(code_block) \
  for (int i = 2; i < SIMULATION.Ny+1-2; i++) { \
    for (int j = 1; j < SIMULATION.Nx-1; j++) { \
      for (int k = 1; k < SIMULATION.Nz-1; k++) { \
        code_block \
      } \
    } \
  }


#define FOR_EACH_3D_NON_BOUNDARY_W_FACE(code_block) \
  for (int i = 1; i < SIMULATION.Ny-1; i++) { \
    for (int j = 1; j < SIMULATION.Nx-1; j++) { \
      for (int k = 2; k < SIMULATION.Nz+1-2; k++) { \
        code_block \
      } \
    } \
  }


#define FOR_EACH_3D_NON_BOUNDARY_CELL(code_block) \
  for (int i = 1; i < SIMULATION.Ny-1; i++) { \
    for (int j = 1; j < SIMULATION.Nx-1; j++) { \
      for (int k = 1; k < SIMULATION.Nz-1; k++) { \
        code_block \
      } \
    } \
  }



MAC::MAC()
{
    // std::cout << "MAC Grid created -  Remember to call InitializeGrid()\n" ;
}

void MAC::InitializeGrid(Domain omega, bool is2D_ )
{
    if(is2D_ == false){
    
    is2D = false;
    this->omega = omega;
    double min = omega.xf - omega.x0;
    this->Nx = SIMULATION.GRID_SIZE;
    this->Ny = -1;
    this->Nz = -1;
    if ((omega.yf - omega.y0) < min)
    {
        min = omega.yf - omega.y0;
        this->Nx = -1;
        this->Ny = SIMULATION.GRID_SIZE;
        this->Nz = -1;
    }
    // if((omega.zf - omega.z0) < min){
    //     min = omega.zf - omega.z0;
    //     this->Nx = -1;
    //     this->Ny = -1;
    //     this->Nz = GRID_SIZE;
    // }

    this->dh = (min / (double)SIMULATION.GRID_SIZE);

    this->Nx = (int)((omega.xf - omega.x0) / dh);
    this->Ny = (int)((omega.yf - omega.y0) / dh);
    this->Nz = (int)((omega.zf - omega.z0) / dh);

    // std::cout << "MAC Grid initialized -  Parameters\n-h = " << std::to_string(this->dh) << "\n-Nx = " << std::to_string(this->Nx) <<" \n-Ny = " << std::to_string(this->Ny)
    //<< "\n-Nz = " << std::to_string(this->Nz) << std::endl;

    // allocating memory

    this->u = VectorXd(((Nx + 1) * Ny * Nz));        //(double*)calloc(((Nx+1) * Ny * Nz),sizeof(double));
    this->v = VectorXd((Nx * (Ny + 1) * Nz));        //(double*)calloc((Nx * (Ny+1) * Nz),sizeof(double));
    this->w = VectorXd((Nx * (Ny) * (Nz + 1)));      //(double*)calloc((Nx * (Ny) * (Nz+1)),sizeof(double));
    this->p = VectorXd((Nx * (Ny) * (Nz)));          //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->SOLID_MASK = VectorXd((Nx * (Ny) * (Nz))); //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->U_UPDATE_MASK = VectorXd(((Nx + 1) * Ny * Nz));
    this->V_UPDATE_MASK = VectorXd(((Nx) * (Ny + 1) * Nz));
    this->W_UPDATE_MASK = VectorXd(((Nx)*Ny * (Nz + 1)));

    this->u.setZero();
    this->v.setZero();
    this->w.setZero();
    this->p.setZero();
    this->SOLID_MASK.setZero();

    this->W_UPDATE_MASK.setZero();
    this->V_UPDATE_MASK.setZero();
    this->W_UPDATE_MASK.setZero();
    }
    else{
        this->is2D = true;
        Domain2D omega2d = Domain2D();
        omega2d.x0 = omega.x0;
        omega2d.xf = omega.xf;
        omega2d.y0 = omega.y0;
        omega2d.yf = omega.yf;


        InitializeGrid(omega2d);
    }
}

void MAC::SetLevelGeometry(int (*SolidMaskFunction)(int, int, int))
{
    if(is2D){
        std::cout << "ERROR - MISMATCHED FUNCTION CALLED ON SET LEVEL GEOMETRY" << std::endl;
    }
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {

                this->SetSolid(i, j, k, SolidMaskFunction(i, j, k));
            }
        }
    }

    // update the component update vector with information
    // if a face sits at a fluid-fluid, it needs projectioon update and it must be solved normally (FLUID_CELL)
    // if a face sits at fluid-solid/inflow or solid/inflow-fluid, it isn't solved, and doesnt need projection update (SOLID_CELL OR INFLOW)
    // if a face sits at a fluid-empty or empty fluid, it isnt solved (its value is extrapolated by neumman), but it needds projection update (EMPTY_CELL)
    // if a face sits at the vert edge of the domainn, it is a invalid cell and shoulnd be solved (SOLID_CELL)
    // if it is anything else, it shoulnd be here

    //u mask
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                if (j == 0 || j == Nx || i == 0 || i == Ny - 1 || k == 0 || k == Nz - 1)
                {
                    this->SetU_Update_Mask(i, j, k, SOLID_CELL);
                    continue;
                }
                if (this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j - 1, k) == FLUID_CELL)
                {
                    this->SetU_Update_Mask(i, j, k, FLUID_CELL);
                }

                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j - 1, k) == SOLID_CELL) || (this->GetSolid(i, j, k) == SOLID_CELL && this->GetSolid(i, j - 1, k) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j, k, SOLID_CELL);
                }

                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j - 1, k) == INFLOW_CELL) || (this->GetSolid(i, j, k) == INFLOW_CELL && this->GetSolid(i, j - 1, k) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j, k, INFLOW_CELL);
                }
                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j - 1, k) == EMPTY_CELL) || (this->GetSolid(i, j, k) == EMPTY_CELL && this->GetSolid(i, j - 1, k) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j, k, EMPTY_CELL);
                }
            }
        }
    }

    for (int i = 0; i < Ny+1; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                if (j == 0 || j == Nx-1|| i == 0 || i == Ny || k == 0 || k == Nz - 1)
                {
                    this->SetV_Update_Mask(i, j, k, SOLID_CELL);
                    continue;
                }
                if (this->GetSolid(i-1, j, k) == FLUID_CELL && this->GetSolid(i, j, k) == FLUID_CELL)
                {
                    this->SetV_Update_Mask(i, j, k, FLUID_CELL);
                }

                if ((this->GetSolid(i-1, j, k) == FLUID_CELL && this->GetSolid(i, j, k) == SOLID_CELL) || (this->GetSolid(i, j, k) == SOLID_CELL && this->GetSolid(i-1, j, k) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, k, SOLID_CELL);
                }

                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i-1, j, k) == INFLOW_CELL) || (this->GetSolid(i, j, k) == INFLOW_CELL && this->GetSolid(i-1, j, k) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, k, INFLOW_CELL);
                }
                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i-1, j, k) == EMPTY_CELL) || (this->GetSolid(i, j, k) == EMPTY_CELL && this->GetSolid(i-1, j, k) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, k, EMPTY_CELL);
                }
            }
        }
    }

    

    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            for (int k = 0; k < Nz+1; k++)
            {
                if (j == 0 || j == Nx-1|| i == 0 || i == Ny-1 || k == 0 || k == Nz)
                {
                    this->SetW_Update_Mask(i, j, k, SOLID_CELL);
                    continue;
                }
                if (this->GetSolid(i, j, k-1) == FLUID_CELL && this->GetSolid(i, j, k) == FLUID_CELL)
                {
                    this->SetW_Update_Mask(i, j, k, FLUID_CELL);
                }

                if ((this->GetSolid(i, j, k-1) == FLUID_CELL && this->GetSolid(i, j, k) == SOLID_CELL) || (this->GetSolid(i, j, k) == SOLID_CELL && this->GetSolid(i, j, k-1) == FLUID_CELL))
                {
                    this->SetW_Update_Mask(i, j, k, SOLID_CELL);
                }

                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j, k-1) == INFLOW_CELL) || (this->GetSolid(i, j, k) == INFLOW_CELL && this->GetSolid(i, j, k-1) == FLUID_CELL))
                {
                    this->SetW_Update_Mask(i, j, k, INFLOW_CELL);
                }
                if ((this->GetSolid(i, j, k) == FLUID_CELL && this->GetSolid(i, j, k-1) == EMPTY_CELL) || (this->GetSolid(i, j, k) == EMPTY_CELL && this->GetSolid(i, j, k-1) == FLUID_CELL))
                {
                    this->SetW_Update_Mask(i, j, k, EMPTY_CELL);
                }
            }
        }
    }

}


void MAC::SetGrid(Vec3 (*VelocityFunction)(double, double, double, double), double (*PressureFunction)(double, double, double, double), double time)
{    if(is2D){
        std::cout << "ERROR - MISMATCHED FUNCTION CALLED ON SET GRID" << std::endl;
    }
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {

                SetP(i, j, k, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, time));
            }
        }
    }
    // u
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            for (int k = 0; k < Nz; k++)
            {

                SetU(i, j, k, VelocityFunction(j * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, time).u);
            }
        }
    }

    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {

                SetV(i, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, k * dh + dh / 2.0, time).v);
            }
        }
    }

    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz + 1; k++)
            {

                SetW(i, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh, time).w);
            }
        }
    }
}

void MAC::SetBorder(Vec3 (*VelocityFunction)(double, double, double, double), double (*PressureFunction)(double, double, double, double), double t)
{    if(is2D){
        std::cout << "ERROR - MISMATCHED FUNCTION CALLED ON SET BORDER" << std::endl;
    }
    // u z face
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetU(i, j, 0, VelocityFunction(j * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, 0 * dh + dh / 2.0, t).u);
            this->SetU(i, j, Nz - 1, VelocityFunction(j * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, (Nz - 1) * dh + dh / 2.0, t).u);
        }
    }

    // u y face
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            this->SetU(0, j, k, VelocityFunction(j * dh - this->omega.x0, 0 * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);
            this->SetU(Ny - 1, j, k, VelocityFunction(j * dh - this->omega.x0, (Ny - 1) * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);
        }
    }
    // u x face
    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz; k++)
        {
            this->SetU(i, 0, k, VelocityFunction(0 * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);
            this->SetU(i, 1, k, VelocityFunction(1 * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);

            this->SetU(i, Nx + 1 - 1, k, VelocityFunction((Nx + 1 - 1) * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);
            this->SetU(i, Nx + 1 - 2, k, VelocityFunction((Nx + 1 - 2) * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t).u);
        }
    }

    // v faces
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny + 1; i++)
        {
            this->SetV(i, j, 0, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, 0 * dh + dh / 2.0, t).v);
            this->SetV(i, j, Nz - 1, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, (Nz - 1) * dh + dh / 2.0, t).v);
        }
    }

    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Ny + 1; i++)
        {
            this->SetV(i, 0, k, VelocityFunction(0 * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, k * dh + dh / 2.0, t).v);
            this->SetV(i, Nx - 1, k, VelocityFunction((Nx - 1) * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, k * dh + dh / 2.0, t).v);
        }
    }

    for (int k = 0; k < Nz; k++)
    {
        for (int j = 0; j < Nx; j++)
        {
            this->SetV(0, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, 0 * dh - this->omega.y0, k * dh + dh / 2.0, t).v);
            this->SetV(Ny + 1 - 1, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny + 1 - 1) * dh - this->omega.y0, k * dh + dh / 2.0, t).v);

            this->SetV(1, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, 1 * dh - this->omega.y0, k * dh + dh / 2.0, t).v);
            this->SetV(Ny + 1 - 2, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny + 1 - 2) * dh - this->omega.y0, k * dh + dh / 2.0, t).v);
        }
    }

    // w face

    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetW(i, j, 0, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, 0 * dh, t).w);
            this->SetW(i, j, Nz + 1 - 1, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, (Nz + 1 - 1) * dh, t).w);

            this->SetW(i, j, 1, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, 1 * dh, t).w);
            this->SetW(i, j, Nz + 1 - 2, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, (Nz + 1 - 2) * dh, t).w);
        }
    }

    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            this->SetW(0, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, 0 * dh + dh / 2.0 - this->omega.y0, k * dh, t).w);
            this->SetW(Ny - 1, j, k, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny - 1) * dh + dh / 2.0 - this->omega.y0, k * dh, t).w);
        }
    }

    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            this->SetW(i, 0, k, VelocityFunction(0 * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh, t).w);
            this->SetW(i, Nx - 1, k, VelocityFunction((Nx - 1) * dh + dh / 2.0 - this->omega.x0, (i)*dh + dh / 2.0 - this->omega.y0, k * dh, t).w);
        }
    }

    // Pressure
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetP(i, j, 0, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, 0 * dh + dh / 2.0, t));
            this->SetP(i, j, Nz - 1, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, (Nz - 1) * dh + dh / 2.0, t));
        }
    }

    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetP(i, 0, k, PressureFunction(0 * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t));
            this->SetP(i, Nx - 1, k, PressureFunction((Nx - 1) * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, (k)*dh + dh / 2.0, t));
        }
    }

    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            this->SetP(0, j, k, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, 0 * dh + dh / 2.0 - this->omega.y0, k * dh + dh / 2.0, t));
            this->SetP(Ny - 1, j, k, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny - 1) * dh + dh / 2.0 - this->omega.y0, (k)*dh + dh / 2.0, t));
        }
    }
}


// Sets the pressure on the step and everything


double MAC::GetDivergencyAt(int i, int j, int k)
{
    if(this->GetSolid(i,j,k) != FLUID_CELL) return 0;
    
    double divU = (this->GetU(i, j + 1, k) - this->GetU(i, j, k)) / this->dh;
    double divV = (this->GetV(i + 1, j, k) - this->GetV(i, j, k)) / this->dh;
    double divW = (this->GetW(i, j, k + 1) - this->GetW(i, j, k)) / this->dh;
    return (divU + divV + divW);
}



// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPxAt(int i, int j, int k)
{

    return (this->GetP(i, j, k) - this->GetP(i, j - 1, k)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPyAt(int i, int j, int k)
{

    return (this->GetP(i, j, k) - this->GetP(i - 1, j, k)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPzAt(int i, int j, int k)
{

    return (this->GetP(i, j, k) - this->GetP(i, j, k - 1)) / this->dh;
};


// double chheck all of these, they will cause problems
double MAC::getVatU(int i, int j, int k)
{
    double top = (this->GetV(i, j - 1, k) + this->GetV(i, j, k)) / 2.0;
    double botton = (this->GetV(i + 1, j - 1, k) + this->GetV(i + 1, j, k)) / 2.0;
    ;

    return (top + botton) / 2.0;
}

double MAC::getWatU(int i, int j, int k)
{
    double foward = (this->GetW(i, j - 1, k + 1) + this->GetW(i, j, k + 1)) / 2.0;
    double backward = (this->GetW(i, j - 1, k) + this->GetW(i, j, k)) / 2.0;

    return (foward + backward) / 2.0;
}

double MAC::getUatV(int i, int j, int k)
{
    double left = (this->GetU(i, j + 1, k) + this->GetU(i - 1, j + 1, k)) / 2.0;
    double right = (this->GetU(i, j, k) + this->GetU(i - 1, j, k)) / 2.0;

    return (left + right) / 2.0;
};
double MAC::getWatV(int i, int j, int k)
{
    double foward = (this->GetW(i, j, k + 1) + this->GetW(i, j + 1, k + 1)) / 2.0;
    double backward = (this->GetW(i, j, k) + this->GetW(i - 1, j, k)) / 2.0;

    return (foward + backward) / 2.0;
};

double MAC::getUatW(int i, int j, int k)
{
    double foward = (this->GetU(i, j + 1, k) + this->GetU(i, j + 1, k - 1)) / 2.0;
    double backward = (this->GetU(i, j, k) + this->GetU(i, j, k - 1)) / 2.0;

    return (foward + backward) / 2.0;
};
double MAC::getVatW(int i, int j, int k)
{
    double top = (this->GetV(i, j, k) + this->GetV(i, j, k - 1)) / 2.0;
    double botton = (this->GetV(i + 1, j, k) + this->GetV(i + 1, j, k - 1)) / 2.0;

    return (top + botton) / 2.0;
};




void MAC::AddAcceleration(Vec3 a,double dt){

    FOR_EACH_3D_NON_BOUNDARY_U_FACE(
        if(this->GetSolid(i,j-1,k) != SOLID_CELL && this->GetSolid(i,j,k) != SOLID_CELL){
                SetU(i, j,k, GetU(i,j,k) + a.u*dt);}
    )
    FOR_EACH_3D_NON_BOUNDARY_V_FACE(
        if(this->GetSolid(i-1,j,k) != SOLID_CELL && this->GetSolid(i,j,k) != SOLID_CELL){
                SetV(i, j,k, GetV(i,j,k) + a.v*dt);}
    )

    FOR_EACH_3D_NON_BOUNDARY_W_FACE(
        if(this->GetSolid(i,j,k-1) != SOLID_CELL && this->GetSolid(i,j,k) != SOLID_CELL){
                SetW(i, j,k, GetW(i,j,k) + a.w*dt);}
    )



}






//======================= 2D OVERLOADS==================


void MAC::InitializeGrid(Domain2D omega)
{
    is2D = true;


    this->omega.x0 = omega.x0;
    this->omega.xf = omega.xf;

    this->omega.y0 = omega.y0;
    this->omega.yf = omega.yf;

    double min = omega.xf - omega.x0;
    this->Nx = SIMULATION.GRID_SIZE;
    this->Ny = -1;

    if ((omega.yf - omega.y0) < min)
    {
        min = omega.yf - omega.y0;
        this->Nx = -1;
        this->Ny = SIMULATION.GRID_SIZE;

    }
    // if((omega.zf - omega.z0) < min){
    //     min = omega.zf - omega.z0;
    //     this->Nx = -1;
    //     this->Ny = -1;
    //     this->Nz = GRID_SIZE;
    // }

    this->dh = (min / (double)SIMULATION.GRID_SIZE);

    this->Nx = (int)((omega.xf - omega.x0) / dh);
    this->Ny = (int)((omega.yf - omega.y0) / dh);


    // std::cout << "MAC Grid initialized -  Parameters\n-h = " << std::to_string(this->dh) << "\n-Nx = " << std::to_string(this->Nx) <<" \n-Ny = " << std::to_string(this->Ny)
    //<< "\n-Nz = " << std::to_string(this->Nz) << std::endl;

    // allocating memory

    this->u = VectorXd(((Nx + 1) * Ny));        //(double*)calloc(((Nx+1) * Ny * Nz),sizeof(double));
    this->v = VectorXd((Nx * (Ny + 1)));        //(double*)calloc((Nx * (Ny+1) * Nz),sizeof(double));

    this->p = VectorXd((Nx * (Ny)));          //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->SOLID_MASK = VectorXd((Nx * (Ny))); //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->U_UPDATE_MASK = VectorXd(((Nx + 1) * Ny ));
    this->V_UPDATE_MASK = VectorXd(((Nx) * (Ny + 1)));


    this->u.setZero();
    this->v.setZero();

    this->p.setZero();
    this->SOLID_MASK.setZero();


    this->V_UPDATE_MASK.setZero();

    this->Nz = 0;


}

void MAC::SetLevelGeometry(int (*SolidMaskFunction)(int, int))
{
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            

                this->SetSolid(i, j, SolidMaskFunction(i, j));
            
        }
    }

    // update the component update vector with information
    // if a face sits at a fluid-fluid, it needs projectioon update and it must be solved normally (FLUID_CELL)
    // if a face sits at fluid-solid/inflow or solid/inflow-fluid, it isn't solved, and doesnt need projection update (SOLID_CELL OR INFLOW)
    // if a face sits at a fluid-empty or empty fluid, it isnt solved (its value is extrapolated by neumman), but it needds projection update (EMPTY_CELL)
    // if a face sits at the vert edge of the domainn, it is a invalid cell and shoulnd be solved (SOLID_CELL)
    // if it is anything else, it shoulnd be here

    //u mask
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {

                if (j == 0 || j == Nx || i == 0 || i == Ny - 1)
                {
                    this->SetU_Update_Mask(i, j,SOLID_CELL);
                    continue;
                }
                if (this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i, j - 1) == FLUID_CELL)
                {
                    this->SetU_Update_Mask(i, j, FLUID_CELL);
                }

                if ((this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i, j - 1) == SOLID_CELL) || (this->GetSolid(i, j) == SOLID_CELL && this->GetSolid(i, j - 1) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j, SOLID_CELL);
                }

                if ((this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i, j - 1) == INFLOW_CELL) || (this->GetSolid(i, j) == INFLOW_CELL && this->GetSolid(i, j - 1) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j, INFLOW_CELL);
                }
                if ((this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i, j - 1) == EMPTY_CELL) || (this->GetSolid(i, j) == EMPTY_CELL && this->GetSolid(i, j - 1) == FLUID_CELL))
                {
                    this->SetU_Update_Mask(i, j,  EMPTY_CELL);
                }
            
        }
    }

    for (int i = 0; i < Ny+1; i++)
    {
        for (int j = 0; j < Nx ; j++)
        {
            
                if (j == 0 || j == Nx-1|| i == 0 || i == Ny)
                {
                    this->SetV_Update_Mask(i, j,  SOLID_CELL);
                    continue;
                }
                if (this->GetSolid(i-1, j) == FLUID_CELL && this->GetSolid(i, j) == FLUID_CELL)
                {
                    this->SetV_Update_Mask(i, j, FLUID_CELL);
                }

                if ((this->GetSolid(i-1, j) == FLUID_CELL && this->GetSolid(i, j) == SOLID_CELL) || (this->GetSolid(i, j) == SOLID_CELL && this->GetSolid(i-1, j) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, SOLID_CELL);
                }

                if ((this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i-1, j) == INFLOW_CELL) || (this->GetSolid(i, j) == INFLOW_CELL && this->GetSolid(i-1, j) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, INFLOW_CELL);
                }
                if ((this->GetSolid(i, j) == FLUID_CELL && this->GetSolid(i-1, j) == EMPTY_CELL) || (this->GetSolid(i, j) == EMPTY_CELL && this->GetSolid(i-1, j) == FLUID_CELL))
                {
                    this->SetV_Update_Mask(i, j, EMPTY_CELL);
                }
            
        }
    }

    

   

}

void MAC::SetGrid(Vec2 (*VelocityFunction)(double, double, double), double (*PressureFunction)(double, double, double), double time)
{
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
           

                SetP(i, j,PressureFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, time));
            
        }
    }
    // u
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            

                SetU(i, j, VelocityFunction(j * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, time).u);
            
        }
    }

    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            

                SetV(i, j, VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, time).v);
            
        }
    }

}


void MAC::SetBorder(Vec2 (*VelocityFunction)(double, double, double), double (*PressureFunction)(double, double, double), double t)
{


    // u y face
    for (int j = 0; j < Nx + 1; j++)
    {

        this->SetU(0, j, VelocityFunction(j * dh - this->omega.x0, 0 * dh + dh / 2.0 - this->omega.y0, t).u);
        this->SetU(Ny - 1, j, VelocityFunction(j * dh - this->omega.x0, (Ny - 1) * dh + dh / 2.0 - this->omega.y0, t).u);
        
    }
    // u x face
    for (int i = 0; i < Ny; i++)
    {

        this->SetU(i, 0,  VelocityFunction(0 * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0,  t).u);
        this->SetU(i, 1,  VelocityFunction(1 * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0,  t).u);
        this->SetU(i, Nx + 1 - 1,  VelocityFunction((Nx + 1 - 1) * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0,  t).u);
        this->SetU(i, Nx + 1 - 2, VelocityFunction((Nx + 1 - 2) * dh - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0,  t).u);
        
    }




    for (int i = 0; i < Ny + 1; i++)
    {
        this->SetV(i, 0,  VelocityFunction(0 * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, t).v);
        this->SetV(i, Nx - 1, VelocityFunction((Nx - 1) * dh + dh / 2.0 - this->omega.x0, i * dh - this->omega.y0, t).v);
    }
    


    for (int j = 0; j < Nx; j++)
    {
        this->SetV(0, j,  VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, 0 * dh - this->omega.y0, t).v);
        this->SetV(Ny + 1 - 1, j,  VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny + 1 - 1) * dh - this->omega.y0,  t).v);
        this->SetV(1, j,  VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, 1 * dh - this->omega.y0, t).v);
        this->SetV(Ny + 1 - 2, j,  VelocityFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny + 1 - 2) * dh - this->omega.y0, t).v);
    }
    

    // w face

 

    // Pressure


    for (int i = 0; i < Ny; i++)
    {
        this->SetP(i, 0,PressureFunction(0 * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0, t));
        this->SetP(i, Nx - 1, PressureFunction((Nx - 1) * dh + dh / 2.0 - this->omega.x0, i * dh + dh / 2.0 - this->omega.y0,  t));
    }
    

    for (int j = 0; j < Nx; j++)
    {

            this->SetP(0, j, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, 0 * dh + dh / 2.0 - this->omega.y0, t));
            this->SetP(Ny - 1, j, PressureFunction(j * dh + dh / 2.0 - this->omega.x0, (Ny - 1) * dh + dh / 2.0 - this->omega.y0, t));

    }
}

// sets a homogeneous neumman boundary condition at the exit of the x domain, for the canal experiments

// Sets the pressure on the step and everything


double MAC::GetDivergencyAt(int i, int j)
{    if(this->GetSolid(i,j) != FLUID_CELL) return 0;
    double divU = (this->GetU(i, j + 1) - this->GetU(i, j)) / this->dh;
    double divV = (this->GetV(i + 1, j) - this->GetV(i, j)) / this->dh;
    return (divU + divV);
}

// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPxAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i, j - 1)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPyAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i - 1, j)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC::GetGradPzAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i, j)) / this->dh;
};





void MAC::AddAcceleration(Vec2 a,double dt){
    //only add to fluid faces
    FOR_EACH_2D_NON_BOUNDARY_U_FACE(
            if((this->GetSolid(i,j-1) == FLUID_CELL ||this->GetSolid(i,j-1) == EMPTY_CELL) && (this->GetSolid(i,j) == FLUID_CELL || this->GetSolid(i,j) == EMPTY_CELL)){
                SetU(i, j, GetU(i,j) + a.u*dt);}
            )

    FOR_EACH_2D_NON_BOUNDARY_V_FACE(
            if((this->GetSolid(i-1,j) == FLUID_CELL || this->GetSolid(i-1,j) == EMPTY_CELL)&& (this->GetSolid(i,j) == FLUID_CELL || this->GetSolid(i,j) == EMPTY_CELL)){
                SetV(i, j, GetV(i,j) + a.v*dt);
            }
    )    
        

}




void MAC::ResetFluidCells(){

    if(is2D){
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            if(this->GetSolid(i,j) != SOLID_CELL ){
                this->SetSolid(i,j,EMPTY_CELL); //resets all fluid cells to empty, keeps the solid mask intact and it is more efficient
            }

        }
    }
    }
    else{
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                for(int k = 0;k<Nz;k++){
                if(this->GetSolid(i,j,k) != SOLID_CELL ){
                    this->SetSolid(i,j,k,EMPTY_CELL); //resets all fluid cells to empty, keeps the solid mask intact and it is more efficient
                }
                }

            }
        }
    }


}






// double chheck all of these, they will cause problems
double MAC::getVatU(int i, int j)
{
    double top = (this->GetV(i, j - 1) + this->GetV(i, j)) / 2.0;
    double botton = (this->GetV(i + 1, j - 1) + this->GetV(i + 1, j)) / 2.0;
    ;

    return (top + botton) / 2.0;
}



double MAC::getUatV(int i, int j)
{
    double left = (this->GetU(i, j + 1) + this->GetU(i - 1, j + 1)) / 2.0;
    double right = (this->GetU(i, j) + this->GetU(i - 1, j)) / 2.0;

    return (left + right) / 2.0;
};





void MAC::SetNeumannBorder() {
    if(is2D) {
        // 2D case - no k index
        for (int i = 1; i < Ny - 1; i++) {
            this->SetU(i, Nx + 1 - 2, this->GetU(i, Nx + 1 - 3));
            this->SetU(i, Nx + 1 - 1, this->GetU(i, Nx + 1 - 2));
            this->SetV(i,Nx-1,this->GetV(i,Nx-2)); //imposing on v 
        }
        return;
    }
    
    // 3D case
    for (int i = 1; i < Ny - 1; i++) {
        for (int k = 1; k < Nz - 1; k++) {
            this->SetU(i, Nx + 1 - 2, k, this->GetU(i, Nx + 1 - 3, k));
            this->SetU(i, Nx + 1 - 1, k, this->GetU(i, Nx + 1 - 2, k));
            this->SetV(i,Nx-1,k,this->GetV(i,Nx-2,k)); 
            this->SetW(i,Nx-1,k,this->GetW(i,Nx-2,k)); 
        }
    }
}

void MAC::SetNeumannBorderPressure() {
    if(SIMULATION.level == LevelConfiguration::LID_CAVITY || SIMULATION.level == LevelConfiguration::DAMBREAK) {
        if(is2D) {
            // 2D lid cavity - no k index
            for (int j = 0; j < Nx; j++) {
                this->SetP(0, j, this->GetP(1, j));
                this->SetP(Ny - 1, j, this->GetP(Ny - 2, j));
            }
            for (int i = 0; i < Ny; i++) {
                this->SetP(i, 0, this->GetP(i, 1));
                this->SetP(i, Nx - 1, this->GetP(i, Nx - 2));
            }
        } else {
            // 3D lid cavity
            for (int j = 0; j < Nx; j++) {
                for (int k = 0; k < Nz; k++) {
                    this->SetP(0, j, k, this->GetP(1, j, k));
                    this->SetP(Ny - 1, j, k, this->GetP(Ny - 2, j, k));
                }
            }
            for (int k = 0; k < Nz; k++) {
                for (int i = 0; i < Ny; i++) {
                    this->SetP(i, 0, k, this->GetP(i, 1, k));
                    this->SetP(i, Nx - 1, k, this->GetP(i, Nx - 2, k));
                }
            }
            for (int j = 0; j < Nx; j++) {
                for (int i = 0; i < Ny; i++) {
                    this->SetP(i, j, 0, this->GetP(i, j, 1));
                    this->SetP(i, j, Nz - 1, this->GetP(i, j, Nz - 2));
                }
            }
        }
    }
    
    if(SIMULATION.level == LevelConfiguration::STEP || 
       SIMULATION.level == LevelConfiguration::OBSTACLE) {
        if(is2D) {
            // 2D step/obstacle/pipe - no k index
            for (int i = 0; i < Ny; i++) {
                if (this->GetSolid(i, Nx - 1) == EMPTY_CELL) {
                    this->SetP(i, Nx - 1, 0);
                } else {
                    this->SetP(i, Nx - 1, this->GetP(i, Nx - 2));
                }
                this->SetP(i, 0, this->GetP(i, 1));
            }
            for (int j = 0; j < Nx; j++) {
                this->SetP(0, j, this->GetP(1, j));
                this->SetP(Ny - 1, j, this->GetP(Ny - 2, j));
            }
        } else {
            // 3D step/obstacle
            for (int j = 0; j < Nx; j++) {
                for (int i = 0; i < Ny; i++) {
                    this->SetP(i, j, 0, this->GetP(i, j, 1));
                    this->SetP(i, j, Nz - 1, this->GetP(i, j, Nz - 2));
                }
            }
            for (int k = 0; k < Nz; k++) {
                for (int i = 0; i < Ny; i++) {
                    if (this->GetSolid(i, Nx - 1, k) == EMPTY_CELL) {
                        this->SetP(i, Nx - 1, k, 0);
                    } else {
                        this->SetP(i, Nx - 1, k, this->GetP(i, Nx - 2, k));
                    }
                    this->SetP(i, 0, k, this->GetP(i, 1, k));
                }
            }
            for (int j = 0; j < Nx; j++) {
                for (int k = 0; k < Nz; k++) {
                    this->SetP(0, j, k, this->GetP(1, j, k));
                    this->SetP(Ny - 1, j, k, this->GetP(Ny - 2, j, k));
                }
            }
        }
    }
}

void MAC::ExportGrid(int iteration) {
    std::string levelStr = LevelConfigurationToString(SIMULATION.level);
    std::string exportBasePath = "Exports";
    std::string dimStr = is2D ? "_2D" : "_3D";
    std::string exportPath = exportBasePath + "/" + levelStr + "/" + 
                            std::to_string(SIMULATION.GRID_SIZE) + dimStr + 
                            "_re" + std::to_string(int(SIMULATION.RE)) + "/VTK/";
    
    std::filesystem::create_directories(exportPath);
    
    std::ostringstream oss;
    oss << exportPath << "grid-" << std::setw(4) << std::setfill('0') << iteration << ".vti";
    std::string filename = oss.str();
    
    std::ofstream vtiFile(filename);
    if (!vtiFile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    int xExtent = this->Nx - 1;
    int yExtent = this->Ny - 1;
    int zExtent = is2D ? 0 : (this->Nz - 1);
    
    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << "  <ImageData WholeExtent=\"0 " << xExtent << " 0 " << yExtent 
            << " 0 " << zExtent << "\" Spacing=\"" << this->dh << " " 
            << this->dh << " " << (is2D ? 1 : this->dh) << "\">\n";
    vtiFile << "    <Piece Extent=\"0 " << xExtent << " 0 " << yExtent 
            << " 0 " << zExtent << "\">\n";
    vtiFile << "      <PointData>\n";
    
    // Velocity vector field
    vtiFile << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    if(is2D) {
        // 2D case - no k loops
        for (int i = 0; i < this->Ny; ++i) {
            for (int j = 0; j < this->Nx; ++j) {
                float u_val = (j < this->Nx - 1) ? (GetU(i, j) + GetU(i, j + 1)) / 2.0f : 0.0f;
                float v_val = (i < this->Ny - 1) ? (GetV(i, j) + GetV(i + 1, j)) / 2.0f : 0.0f;
                float w_val = 0.0f;
                
                u_val = std::isfinite(u_val) ? u_val : 0.0f;
                v_val = std::isfinite(v_val) ? v_val : 0.0f;
                
                vtiFile << u_val << " " << v_val << " " << w_val << " ";
            }
            vtiFile << "\n";
        }
    } else {
        // 3D case
        for (int k = 0; k < this->Nz; ++k) {
            for (int i = 0; i < this->Ny; ++i) {
                for (int j = 0; j < this->Nx; ++j) {
                    float u_val = (j < this->Nx - 1) ? (GetU(i, j, k) + GetU(i, j + 1, k)) / 2.0f : 0.0f;
                    float v_val = (i < this->Ny - 1) ? (GetV(i, j, k) + GetV(i + 1, j, k)) / 2.0f : 0.0f;
                    float w_val = (k < this->Nz - 1) ? (GetW(i, j, k) + GetW(i, j, k + 1)) / 2.0f : 0.0f;
                    
                    u_val = std::isfinite(u_val) ? u_val : 0.0f;
                    v_val = std::isfinite(v_val) ? v_val : 0.0f;
                    w_val = std::isfinite(w_val) ? w_val : 0.0f;
                    
                    vtiFile << u_val << " " << v_val << " " << w_val << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";
    
    // Pressure field
    vtiFile << "        <DataArray type=\"Float64\" Name=\"Pressure\" format=\"ascii\">\n";
    if(is2D) {
        for (int i = 0; i < this->Ny; ++i) {
            for (int j = 0; j < this->Nx; ++j) {
                vtiFile << this->GetP(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < this->Nz; ++k) {
            for (int i = 0; i < this->Ny; ++i) {
                for (int j = 0; j < this->Nx; ++j) {
                    vtiFile << this->GetP(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";
    
    // Divergence field
    vtiFile << "        <DataArray type=\"Float64\" Name=\"Divergence\" format=\"ascii\">\n";
    if(is2D) {
        for (int i = 0; i < this->Ny; ++i) {
            for (int j = 0; j < this->Nx; ++j) {
                vtiFile << this->GetDivergencyAt(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < this->Nz; ++k) {
            for (int i = 0; i < this->Ny; ++i) {
                for (int j = 0; j < this->Nx; ++j) {
                    vtiFile << this->GetDivergencyAt(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";
    
    // Solid field
    vtiFile << "        <DataArray type=\"Int32\" Name=\"Solid\" format=\"ascii\">\n";
    if(is2D) {
        for (int i = 0; i < this->Ny; ++i) {
            for (int j = 0; j < this->Nx; ++j) {
                vtiFile << this->GetSolid(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < this->Nz; ++k) {
            for (int i = 0; i < this->Ny; ++i) {
                for (int j = 0; j < this->Nx; ++j) {
                    vtiFile << this->GetSolid(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";
    
    vtiFile << "      </PointData>\n";
    vtiFile << "    </Piece>\n";
    vtiFile << "  </ImageData>\n";
    vtiFile << "</VTKFile>\n";
    
    vtiFile.close();
}

double MAC::MaxAbsoluteDifference(MAC &grid) {
    double max = 0.0;
    
    if(is2D) {
        // 2D case - no k index
        max = abs(this->GetU(0, 0) - grid.GetU(0, 0));
        
        // Check U component
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx + 1; j++) {
                if (max < abs(this->GetU(i, j) - grid.GetU(i, j))) {
                    max = abs(this->GetU(i, j) - grid.GetU(i, j));
                }
            }
        }
        
        // Check V component
        for (int i = 0; i < this->Ny + 1; i++) {
            for (int j = 0; j < this->Nx; j++) {
                if (max < abs(this->GetV(i, j) - grid.GetV(i, j))) {
                    max = abs(this->GetV(i, j) - grid.GetV(i, j));
                }
            }
        }
    } else {
        // 3D case
        max = abs(this->GetU(0, 0, 0) - grid.GetU(0, 0, 0));
        
        // Check U component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx + 1; j++) {
                    if (max < abs(this->GetU(i, j, k) - grid.GetU(i, j, k))) {
                        max = abs(this->GetU(i, j, k) - grid.GetU(i, j, k));
                    }
                }
            }
        }
        
        // Check V component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny + 1; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    if (max < abs(this->GetV(i, j, k) - grid.GetV(i, j, k))) {
                        max = abs(this->GetV(i, j, k) - grid.GetV(i, j, k));
                    }
                }
            }
        }
        
        // Check W component
        for (int k = 0; k < this->Nz + 1; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    if (max < abs(this->GetW(i, j, k) - grid.GetW(i, j, k))) {
                        max = abs(this->GetW(i, j, k) - grid.GetW(i, j, k));
                    }
                }
            }
        }
    }
    
    return max;
}

double MAC::MaxAbsoluteDifferencePressure(MAC &grid) {
    double max = 0.0;
    
    if(is2D) {
        // 2D case - no k index
        max = abs(this->GetP(0, 0) - grid.GetP(0, 0));
        
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx; j++) {
                if (max < abs(this->GetP(i, j) - grid.GetP(i, j))) {
                    max = abs(this->GetP(i, j) - grid.GetP(i, j));
                }
            }
        }
    } else {
        // 3D case
        max = abs(this->GetP(0, 0, 0) - grid.GetP(0, 0, 0));
        
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    if (max < abs(this->GetP(i, j, k) - grid.GetP(i, j, k))) {
                        max = abs(this->GetP(i, j, k) - grid.GetP(i, j, k));
                    }
                }
            }
        }
    }
    
    return max;
}

double MAC::GetMaxVelocity() {
    double mag = 0.0;
    
    if(is2D) {
        // 2D case - no k index
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx; j++) {
                float u_val = (j < this->Nx - 1) ? 
                    (GetU(i, j) + GetU(i, j + 1)) / 2.0f : 0.0f;
                float v_val = (i < this->Ny - 1) ? 
                    (GetV(i, j) + GetV(i + 1, j)) / 2.0f : 0.0f;
                
                double current_mag = sqrt(u_val * u_val + v_val * v_val);
                
                if (current_mag > mag) {
                    mag = current_mag;
                }
            }
        }
    } else {
        // 3D case
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    float u_val = (j < this->Nx - 1) ? 
                        (GetU(i, j, k) + GetU(i, j + 1, k)) / 2.0f : 0.0f;
                    float v_val = (i < this->Ny - 1) ? 
                        (GetV(i, j, k) + GetV(i + 1, j, k)) / 2.0f : 0.0f;
                    float w_val = (k < this->Nz - 1) ? 
                        (GetW(i, j, k) + GetW(i, j, k + 1)) / 2.0f : 0.0f;
                    
                    double current_mag = sqrt(u_val * u_val + v_val * v_val + w_val * w_val);
                    
                    if (current_mag > mag) {
                        mag = current_mag;
                    }
                }
            }
        }
    }
    
    return mag;
}

double MAC::GetDivSum() {
    double value = 0.0;
    
    if(is2D) {
        // 2D case - no k index
        for (int i = 1; i < Ny - 1; i++) {
            for (int j = 1; j < Nx - 1; j++) {
                value += abs(GetDivergencyAt(i, j));
            }
        }
    } else {
        // 3D case
        for (int k = 1; k < Nz - 1; k++) {
            for (int i = 1; i < Ny - 1; i++) {
                for (int j = 1; j < Nx - 1; j++) {
                    value += abs(GetDivergencyAt(i, j, k));
                }
            }
        }
    }
    
    return value;
}

int MAC::GetFluidCellCount() {
    int c = 0;
    
    if(is2D) {
        // 2D case - no k index
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                if(this->GetSolid(i, j) == FLUID_CELL) {
                    c++;
                }
            }
        }
    } else {
        // 3D case
        for (int k = 0; k < Nz; k++) {
            for (int i = 0; i < Ny; i++) {
                for (int j = 0; j < Nx; j++) {
                    if(this->GetSolid(i, j, k) == FLUID_CELL) {
                        c++;
                    }
                }
            }
        }
    }
    
    return c;
}

void MAC::CopyGrid(MAC &grid) {
    if(is2D) {
        if(!grid.is2D){
            std::cout << "ERROR - MISMATCHE GRID COPY OPERATION (3D COPIED IN 2D)";
            return;
        }
        // 2D case - no k index
        
        // Copy pressure and solid mask
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx; j++) {
                this->SetP(i, j, grid.GetP(i, j));
                this->SetSolid(i, j, grid.GetSolid(i, j));
            }
        }
        
        // Copy U component
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx + 1; j++) {
                this->SetU(i, j, grid.GetU(i, j));
            }
        }
        
        // Copy V component
        for (int i = 0; i < this->Ny + 1; i++) {
            for (int j = 0; j < this->Nx; j++) {
                this->SetV(i, j, grid.GetV(i, j));
            }
        }
    } else {
        // 3D case
        if(grid.is2D){
            std::cout << "ERROR - MISMATCHE GRID COPY OPERATION (2D COPIED IN 3D)";
        }
        
        // Copy pressure and solid mask
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetP(i, j, k, grid.GetP(i, j, k));
                    this->SetSolid(i, j, k, grid.GetSolid(i, j, k));
                }
            }
        }
        
        // Copy U component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx + 1; j++) {
                    this->SetU(i, j, k, grid.GetU(i, j, k));
                }
            }
        }
        
        // Copy V component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny + 1; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetV(i, j, k, grid.GetV(i, j, k));
                }
            }
        }
        
        // Copy W component
        for (int k = 0; k < this->Nz + 1; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetW(i, j, k, grid.GetW(i, j, k));
                }
            }
        }
    }
}

void MAC::DestroyGrid() {
    // This function is already empty in both versions
    // If you need to free resources in the future, handle both 2D and 3D here
}