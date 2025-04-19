#include "headers/MAC.h"


MAC::MAC()
{
    // std::cout << "MAC Grid created -  Remember to call InitializeGrid()\n" ;
}

void MAC::InitializeGrid(Domain omega)
{
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

void MAC::SetLevelGeometry(int (*SolidMaskFunction)(int, int, int))
{
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
{
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
{
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

// sets a homogeneous neumman boundary condition at the exit of the x domain, for the canal experiments
void MAC::SetNeumannBorder()
{
    //we only set this if the cell we are is empty, if it is, we set it to the value of the right or left non-empty neighboor
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int k = 1; k < Nz - 1; k++)
        {
            this->SetU(i, Nx + 1 - 2, k, this->GetU(i, Nx + 1 - 3, k));
            this->SetU(i, Nx + 1 - 1, k, this->GetU(i, Nx + 1 - 2, k));
        }
    }

    // for(int i = 2;i<Ny+1-2;i++){
    //     for(int k = 1;k<Nz-1;k++){
    //         this->SetV(i,Nx-1,k,this->GetV(i,Nx-2,k));
    //
    //    }
    //
    //}
    //
    // for(int i = 1;i<Ny-1;i++){
    //    for(int k = 2;k<Nz+1-2;k++){
    //        this->SetW(i,Nx-1,k,this->GetW(i,Nx-2,k));
    //
    //    }
    //
    //}
}
// Sets the pressure on the step and everything
void MAC::SetNeumannBorderPressure()
{
    if(SIMULATION.level == LevelConfiguration::LID_CAVITY){
    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            this->SetP(0, j, k, this->GetP(1, j, k));
            this->SetP(Ny - 1, j, k, this->GetP(Ny - 2, j, k));
        }
    }
    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetP(i, 0, k, this->GetP(i, 1, k));
            this->SetP(i, Nx-1, k, this->GetP(i, Nx-2, k));
        }
    }
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetP(i, j, 0, this->GetP(i, j, 1));
            this->SetP(i, j, Nz - 1, this->GetP(i, j, Nz - 2));
        }
    }
    }


    if(SIMULATION.level == LevelConfiguration::STEP || SIMULATION.level == LevelConfiguration::OBSTACLE){
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            this->SetP(i, j, 0, this->GetP(i, j, 1));
            this->SetP(i, j, Nz - 1, this->GetP(i, j, Nz - 2));
        }
    }

    // in this face
    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Ny; i++)
        {
            if (this->GetSolid(i, Nx - 1, k) == EMPTY_CELL)
            {
                this->SetP(i, Nx - 1, k, 0);
            }
            else
            {
                this->SetP(i, Nx - 1, k, this->GetP(i, Nx - 2, k));
            }
            this->SetP(i, 0, k, this->GetP(i, 1, k));
        }
    }

    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            this->SetP(0, j, k, this->GetP(1, j, k));
            this->SetP(Ny - 1, j, k, this->GetP(Ny - 2, j, k));
        }
    }
}

    
}

double MAC::GetDivergencyAt(int i, int j, int k)
{
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

void MAC::DestroyGrid()
{
}

void MAC::CopyGrid(MAC &grid)
{
    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {

                this->SetP(i, j, k, grid.GetP(i, j, k));
                this->SetSolid(i,j,k,grid.GetSolid(i,j,k));
            }
        }
    }

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx + 1; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {

                this->SetU(i, j, k, grid.GetU(i, j, k));
            }
        }
    }
    for (int i = 0; i < this->Ny + 1; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {

                this->SetV(i, j, k, grid.GetV(i, j, k));
            }
        }
    }

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz + 1; k++)
            {

                this->SetW(i, j, k, grid.GetW(i, j, k));
            }
        }
    }
}

double MAC::MaxAbsoluteDifference(MAC &grid)
{
    double max = abs(this->GetU(0, 0, 0) - grid.GetU(0, 0, 0));

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx + 1; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {
                if (max < abs(this->GetU(i, j, k) - grid.GetU(i, j, k)))
                {

                    max = abs(this->GetU(i, j, k) - grid.GetU(i, j, k));
                }
            }
        }
    }
    //
    for (int i = 0; i < this->Ny + 1; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {
                if (max < abs(this->GetV(i, j, k) - grid.GetV(i, j, k)))
                {
                    max = abs(this->GetV(i, j, k) - grid.GetV(i, j, k));
                }
            }
        }
    }

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz + 1; k++)
            {
                if (max < abs(this->GetW(i, j, k) - grid.GetW(i, j, k)))
                {
                    max = abs(this->GetW(i, j, k) - grid.GetW(i, j, k));
                }
            }
        }
    }
    //

    return max;
}

void MAC::ExportGrid(int iteration)
{
    std::string levelStr = LevelConfigurationToString(SIMULATION.level);
    std::string exportBasePath = "Exports";
    std::string exportPath = exportBasePath + "/" + levelStr + "/" +
        std::to_string(SIMULATION.GRID_SIZE) + "_re" +
        std::to_string(int(SIMULATION.RE)) + "/VTK/";
    
    // Create the directory structure if it doesn't exist
    std::filesystem::create_directories(exportPath);
    
    // Create the filename with iteration number
    std::ostringstream oss;
    oss << exportPath << "grid-" << std::setw(4) << std::setfill('0') << iteration << ".vti";
    std::string filename = oss.str();

    
    std::ofstream vtiFile(filename);
    if (!vtiFile.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Calculate extent (number of points is Nx, Ny, Nz)
    int xExtent = this->Nx - 1; // Number of cells in X
    int yExtent = this->Ny - 1; // Number of cells in Y
    int zExtent = this->Nz - 1; // Number of cells in Z
    
    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << " <ImageData WholeExtent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\" "
        << "Origin=\"0 0 0\" Spacing=\"" << this->dh << " " << this->dh << " " << this->dh << "\">\n";
    vtiFile << " <Piece Extent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\">\n";
    
    // Point data section
    vtiFile << " <PointData Vectors=\"velocity\">\n";
    
    // Velocity vector field
    vtiFile << " <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    // Loop order matches VTK's expectation (Z slowest, Y middle, X fastest) -> i donnt know about this exactly, but if claude says so...
    for (int k = 0; k < this->Nz; ++k) { // Z dimension (forward/backward)
        for (int j = 0; j < this->Ny; ++j) { // Y dimension (up/down)
            for (int i = 0; i < this->Nx; ++i) { // X dimension (left/right)
                // Get interpolated velocity components at cell centers
                float u_val = (i < this->Nx - 1) ? (GetU(j, i, k) + GetU(j, i + 1, k)) / 2.0f : 0.0f; // X-component
                float v_val = (j < this->Ny - 1) ? (GetV(j, i, k) + GetV(j + 1, i, k)) / 2.0f : 0.0f; // Y-component
                float w_val = (k < this->Nz - 1) ? (GetW(j, i, k) + GetW(j, i, k + 1)) / 2.0f : 0.0f; // Z-component
                
                // Handle potential NaN/infinite values
                u_val = std::isfinite(u_val) ? u_val : 0.0f;
                v_val = std::isfinite(v_val) ? v_val : 0.0f;
                w_val = std::isfinite(w_val) ? w_val : 0.0f;
                
                vtiFile << u_val << " " << v_val << " " << w_val << " ";
            }
            vtiFile << "\n";
        }
    }
    
    vtiFile << " </DataArray>\n";
    
    // Add pressure field to PointData
    vtiFile << " <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (int k = 0; k < this->Nz; ++k) {
        for (int j = 0; j < this->Ny; ++j) {
            for (int i = 0; i < this->Nx; ++i) {
                vtiFile << this->GetP(j, i, k) << " ";
            }
            vtiFile << "\n";
        }
    }
    vtiFile << " </DataArray>\n";
    
    // Add divergence field to PointData
    vtiFile << " <DataArray type=\"Float32\" Name=\"divergence\" format=\"ascii\">\n";
    for (int k = 0; k < this->Nz; ++k) {
        for (int j = 0; j < this->Ny; ++j) {
            for (int i = 0; i < this->Nx; ++i) {
                vtiFile << this->GetDivergencyAt(j, i, k) << " ";
            }
            vtiFile << "\n";
        }
    }
    vtiFile << " </DataArray>\n";
    
    // Add solid field to PointData
    vtiFile << " <DataArray type=\"Int32\" Name=\"solid\" format=\"ascii\">\n";
    for (int k = 0; k < this->Nz; ++k) {
        for (int j = 0; j < this->Ny; ++j) {
            for (int i = 0; i < this->Nx; ++i) {
                vtiFile << this->GetSolid(j, i, k) << " ";
            }
            vtiFile << "\n";
        }
    }
    vtiFile << " </DataArray>\n";
    
    vtiFile << " </PointData>\n";
    vtiFile << " </Piece>\n";
    vtiFile << " </ImageData>\n";
    vtiFile << "</VTKFile>\n";
    
    vtiFile.close();
}

double MAC::MaxAbsoluteDifferencePressure(MAC &grid)
{
    double max = abs(this->GetP(0, 0, 0) - grid.GetP(0, 0, 0));

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {
            for (int k = 0; k < this->Nz; k++)
            {
                if (max < abs(this->GetP(i, j, k) - grid.GetP(i, j, k)))
                {
                    max = abs(this->GetP(i, j, k) - grid.GetP(i, j, k));
                }
            }
        }
    }
    return max;
}

double MAC::GetDivSum()
{
    double value = 0.0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            for (int k = 1; k < Nz - 1; k++)
            {
                // printf("Div at %d,%d,%d = %f\n",i,j,k,GetDivergencyAt(i,j,k));
                value += abs(GetDivergencyAt(i, j, k));
            }
        }
    }
    return value;
}

double MAC::GetMaxVelocity(){
    //so, check this, I asked deepseek to do this function quickly because I was bored
    //wtf is this shit, why it inverted the fucking loops
    //i GAVE him the loop I copied and pasted from the divsum
    double mag = 0.0;
    for (int k = 0; k < this->Nz; ++k) { // Z dimension (forward/backward)
        for (int j = 0; j < this->Ny; ++j) { // Y dimension (up/down)
            for (int i = 0; i < this->Nx; ++i) { // X dimension (left/right)
                // Get interpolated velocity components at cell centers
                float u_val = (i < this->Nx - 1) ? (GetU(j, i, k) + GetU(j, i + 1, k)) / 2.0f : 0.0f; // X-component
                float v_val = (j < this->Ny - 1) ? (GetV(j, i, k) + GetV(j + 1, i, k)) / 2.0f : 0.0f; // Y-component
                float w_val = (k < this->Nz - 1) ? (GetW(j, i, k) + GetW(j, i, k + 1)) / 2.0f : 0.0f; // Z-component
                
                // Calculate magnitude of velocity vector
                double current_mag = sqrt(u_val * u_val + v_val * v_val + w_val * w_val);
                
                // Update maximum magnitude if current magnitude is larger
                if (current_mag > mag) {
                    mag = current_mag;
                }
            }
        }
    }
    return mag;
}


void MAC::ExportGridVTK(int iteration)
{
    std::string levelStr = LevelConfigurationToString(SIMULATION.level);
    std::string exportBasePath = "Exports";
    std::string exportPath = exportBasePath + "/" + levelStr + "/" +
        std::to_string(SIMULATION.GRID_SIZE) + "_re" +
        std::to_string(int(SIMULATION.RE)) + "/VTK/";
    
    // Create the directory structure if it doesn't exist
    std::filesystem::create_directories(exportPath);
    
    // Create the filename with iteration number
    std::ostringstream oss;
    oss << exportPath << "grid-" << std::setw(4) << std::setfill('0') << iteration << ".vti";
    std::cout << oss.str() << std::endl;

    std::ofstream vtiFile(oss.str());
    if (!vtiFile.is_open())
    {
        throw std::runtime_error("Could not open file for writing");
    }

    // Calculate extent (number of points is Nx, Ny, Nz)
    int xExtent = this->Nx - 1; // Number of cells in X
    int yExtent = this->Ny - 1; // Number of cells in Y
    int zExtent = this->Nz - 1; // Number of cells in Z

    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << "  <ImageData WholeExtent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\" "
            << "Origin=\"0 0 0\" Spacing=\"" << this->dh << " " << this->dh << " " << this->dh << "\">\n";
    vtiFile << "  <Piece Extent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\">\n";
    vtiFile << "  <PointData Vectors=\"velocity\">\n";
    vtiFile << "  <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    // Loop order matches VTK's expectation (Z slowest, Y middle, X fastest)
    for (int k = 0; k < this->Nz; ++k)
    { // Z dimension (forward/backward)
        for (int i = 0; i < this->Ny; ++i)
        { // Y dimension (up/down)
            for (int j = 0; j < this->Nx; ++j)
            { // X dimension (left/right)
                // Get interpolated velocity components at cell centers
                float u_val = (j < this->Nx - 1) ? (GetU(i, j, k) + GetU(i, j + 1, k)) / 2.0f : 0.0f; // X-component
                float v_val = (i < this->Ny - 1) ? (GetV(i, j, k) + GetV(i + 1, j, k)) / 2.0f : 0.0f; // Y-component
                float w_val = (k < this->Nz - 1) ? (GetW(i, j, k) + GetW(i, j, k + 1)) / 2.0f : 0.0f; // Z-component

                // Handle potential NaN/infinite values
                u_val = std::isfinite(u_val) ? u_val : 0.0f;
                v_val = std::isfinite(v_val) ? v_val : 0.0f;
                w_val = std::isfinite(w_val) ? w_val : 0.0f;

                vtiFile << u_val << " " << v_val << " " << w_val << " ";
            }
            vtiFile << "\n";
        }
    }

    // VTK Footer
    vtiFile << "  </DataArray>\n";
    vtiFile << "  </PointData>\n";
    vtiFile << "  </Piece>\n";
    vtiFile << "  </ImageData>\n";
    vtiFile << "</VTKFile>\n";
    vtiFile.close();
}
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
