#include "headers/MAC_2D.h"


MAC2D::MAC2D()
{
    // std::cout << "MAC Grid created -  Remember to call InitializeGrid()\n" ;
}

void MAC2D::InitializeGrid(Domain2D omega)
{
    this->omega = omega;
    double min = omega.xf - omega.x0;
    this->Nx = SIMULATION2D.GRID_SIZE;
    this->Ny = -1;

    if ((omega.yf - omega.y0) < min)
    {
        min = omega.yf - omega.y0;
        this->Nx = -1;
        this->Ny = SIMULATION2D.GRID_SIZE;

    }
    // if((omega.zf - omega.z0) < min){
    //     min = omega.zf - omega.z0;
    //     this->Nx = -1;
    //     this->Ny = -1;
    //     this->Nz = GRID_SIZE;
    // }

    this->dh = (min / (double)SIMULATION2D.GRID_SIZE);

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

}

void MAC2D::SetLevelGeometry(int (*SolidMaskFunction)(int, int))
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

void MAC2D::SetGrid(Vec2 (*VelocityFunction)(double, double, double), double (*PressureFunction)(double, double, double), double time)
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

int MAC2D::GetFluidCellCount(){
    int c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
             
            if(this->GetSolid(i,j) == FLUID_CELL){
                c++;
            }
            

        }
    }
    return c;
}


void MAC2D::SetBorder(Vec2 (*VelocityFunction)(double, double, double), double (*PressureFunction)(double, double, double), double t)
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
void MAC2D::SetNeumannBorder()
{
    //we only set this if the cell we are is empty, if it is, we set it to the value of the right or left non-empty neighboor
    for (int i = 1; i < Ny - 1; i++)
    {
        
            this->SetU(i, Nx + 1 - 2,this->GetU(i, Nx + 1 - 3));
            this->SetU(i, Nx + 1 - 1,  this->GetU(i, Nx + 1 - 2));
        
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
void MAC2D::SetNeumannBorderPressure()
{
    if(SIMULATION2D.level == LevelConfiguration::LID_CAVITY){
    for (int j = 0; j < Nx; j++)
    {
       
            this->SetP(0, j, this->GetP(1, j));
            this->SetP(Ny - 1, j, this->GetP(Ny - 2, j));
        
    }
    }


    if(SIMULATION2D.level == LevelConfiguration::STEP || SIMULATION2D.level == LevelConfiguration::OBSTACLE || SIMULATION2D.level == LevelConfiguration::PIPE){


    // in this face

    for (int i = 0; i < Ny; i++)
        {
            if (this->GetSolid(i, Nx - 1) == EMPTY_CELL)
            {
                this->SetP(i, Nx - 1,  0);
            }
            else
            {
                this->SetP(i, Nx - 1,  this->GetP(i, Nx - 2));
            }
            this->SetP(i, 0,  this->GetP(i, 1));
        }
    

    for (int j = 0; j < Nx; j++)
    {
       
            this->SetP(0, j,  this->GetP(1, j ));
            this->SetP(Ny - 1, j,  this->GetP(Ny - 2, j));
        
    }
    
    
    }
}

double MAC2D::GetDivergencyAt(int i, int j)
{
    double divU = (this->GetU(i, j + 1) - this->GetU(i, j)) / this->dh;
    double divV = (this->GetV(i + 1, j) - this->GetV(i, j)) / this->dh;
    return (divU + divV);
}

// be careful to not try to get the gradient on bondary nodes!
double MAC2D::GetGradPxAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i, j - 1)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC2D::GetGradPyAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i - 1, j)) / this->dh;
};
// be careful to not try to get the gradient on bondary nodes!
double MAC2D::GetGradPzAt(int i, int j)
{

    return (this->GetP(i, j) - this->GetP(i, j)) / this->dh;
};

void MAC2D::DestroyGrid()
{
}

void MAC2D::CopyGrid(MAC2D &grid)
{
    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {


                this->SetP(i, j,grid.GetP(i, j));
                this->SetSolid(i,j,grid.GetSolid(i,j));
            
        }
    }

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx + 1; j++)
        {


                this->SetU(i, j, grid.GetU(i, j));

        }
    }
    for (int i = 0; i < this->Ny + 1; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {


                this->SetV(i, j, grid.GetV(i, j));
            
        }
    }


}

double MAC2D::MaxAbsoluteDifference(MAC2D &grid)
{
    double max = abs(this->GetU(0, 0) - grid.GetU(0, 0));

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx + 1; j++)
        {

                if (max < abs(this->GetU(i, j) - grid.GetU(i, j)))
                {

                    max = abs(this->GetU(i, j) - grid.GetU(i, j));
                }
            
        }
    }
    //
    for (int i = 0; i < this->Ny + 1; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {

                if (max < abs(this->GetV(i, j) - grid.GetV(i, j)))
                {
                    max = abs(this->GetV(i, j) - grid.GetV(i, j));
                }
            
        }
    }


    //

    return max;
}

void MAC2D::ExportGrid(int iteration)
{
    std::string levelStr = LevelConfigurationToString(SIMULATION2D.level);
    std::string exportBasePath = "Exports";
    std::string exportPath = exportBasePath + "/" + levelStr + "/" +
        std::to_string(SIMULATION2D.GRID_SIZE) + "_re" +
        std::to_string(int(SIMULATION2D.RE)) + "/VTK/";
    
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
    
    // Calculate extent (number of points is Nx, Ny)
    int xExtent = this->Nx - 1; // Number of cells in X
    int yExtent = this->Ny - 1; // Number of cells in Y
    
    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << " <ImageData WholeExtent=\"0 " << xExtent << " 0 " << yExtent << " 0 0\" "
        << "Origin=\"0 0 0\" Spacing=\"" << this->dh << " " << this->dh << " 1\">\n";
    vtiFile << " <Piece Extent=\"0 " << xExtent << " 0 " << yExtent << " 0 0\">\n";
    
    // Point data section
    vtiFile << " <PointData Vectors=\"velocity\">\n";
    
    // Velocity vector field
    vtiFile << " <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    // Loop order matches VTK's expectation (Y middle, X fastest)
    for (int j = 0; j < this->Ny; ++j) { // Y dimension (up/down)
        for (int i = 0; i < this->Nx; ++i) { // X dimension (left/right)
            // Get interpolated velocity components at cell centers
            float u_val = (i < this->Nx - 1) ? (GetU(j, i) + GetU(j, i + 1)) / 2.0f : 0.0f; // X-component
            float v_val = (j < this->Ny - 1) ? (GetV(j, i) + GetV(j + 1, i)) / 2.0f : 0.0f; // Y-component
            float w_val = 0.0f; // Z-component is always 0 in 2D
            
            // Handle potential NaN/infinite values
            u_val = std::isfinite(u_val) ? u_val : 0.0f;
            v_val = std::isfinite(v_val) ? v_val : 0.0f;
            
            vtiFile << u_val << " " << v_val << " " << w_val << " ";
        }
        vtiFile << "\n";
    }
    
    vtiFile << " </DataArray>\n";
    
    // Add pressure field to PointData
    vtiFile << " <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (int j = 0; j < this->Ny; ++j) {
        for (int i = 0; i < this->Nx; ++i) {
            vtiFile << this->GetP(j, i) << " ";
        }
        vtiFile << "\n";
    }
    vtiFile << " </DataArray>\n";
    
    // Add divergence field to PointData
    vtiFile << " <DataArray type=\"Float32\" Name=\"divergence\" format=\"ascii\">\n";
    for (int j = 0; j < this->Ny; ++j) {
        for (int i = 0; i < this->Nx; ++i) {
            vtiFile << this->GetDivergencyAt(j, i) << " ";
        }
        vtiFile << "\n";
    }
    vtiFile << " </DataArray>\n";
    
    // Add solid field to PointData
    vtiFile << " <DataArray type=\"Int32\" Name=\"solid\" format=\"ascii\">\n";
    for (int j = 0; j < this->Ny; ++j) {
        for (int i = 0; i < this->Nx; ++i) {
            vtiFile << this->GetSolid(j, i) << " ";
        }
        vtiFile << "\n";
    }
    vtiFile << " </DataArray>\n";
    
    vtiFile << " </PointData>\n";
    vtiFile << " </Piece>\n";
    vtiFile << " </ImageData>\n";
    vtiFile << "</VTKFile>\n";
    
    vtiFile.close();
}


double MAC2D::MaxAbsoluteDifferencePressure(MAC2D &grid)
{
    double max = abs(this->GetP(0, 0) - grid.GetP(0, 0));

    for (int i = 0; i < this->Ny; i++)
    {
        for (int j = 0; j < this->Nx; j++)
        {

                if (max < abs(this->GetP(i, j) - grid.GetP(i, j)))
                {
                    max = abs(this->GetP(i, j) - grid.GetP(i, j));
                }
            
        }
    }
    return max;
}

double MAC2D::GetDivSum()
{
    double value = 0.0;
    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {

                // printf("Div at %d,%d,%d = %f\n",i,j,k,GetDivergencyAt(i,j,k));
                value += abs(GetDivergencyAt(i, j));
            
        }
    }
    return value;
}

double MAC2D::GetMaxVelocity(){
    //so, check this, I asked deepseek to do this function quickly because I was bored
    //wtf is this shit, why it inverted the fucking loops
    //i GAVE him the loop I copied and pasted from the divsum
    double mag = 0.0;
    for (int k = 0; k < 4; ++k) { // Z dimension (forward/backward)
        for (int j = 0; j < this->Ny; ++j) { // Y dimension (up/down)
            for (int i = 0; i < this->Nx; ++i) { // X dimension (left/right)
                // Get interpolated velocity components at cell centers
                float u_val = (i < this->Nx - 1) ? (GetU(j, i) + GetU(j, i + 1)) / 2.0f : 0.0f; // X-component
                float v_val = (j < this->Ny - 1) ? (GetV(j, i) + GetV(j + 1, i)) / 2.0f : 0.0f; // Y-component
                float w_val = 0;
                
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


void MAC2D::ExportGridVTK(int iteration)
{
    std::string levelStr = LevelConfigurationToString(SIMULATION2D.level);
    std::string exportBasePath = "Exports";
    std::string exportPath = exportBasePath + "/" + levelStr + "/" +
        std::to_string(SIMULATION2D.GRID_SIZE) + "_re" +
        std::to_string(int(SIMULATION2D.RE)) + "/VTK/";
    
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
    int zExtent = 4; // Number of cells in Z

    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << "  <ImageData WholeExtent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\" "
            << "Origin=\"0 0 0\" Spacing=\"" << this->dh << " " << this->dh << " " << this->dh << "\">\n";
    vtiFile << "  <Piece Extent=\"0 " << xExtent << " 0 " << yExtent << " 0 " << zExtent << "\">\n";
    vtiFile << "  <PointData Vectors=\"velocity\">\n";
    vtiFile << "  <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    // Loop order matches VTK's expectation (Z slowest, Y middle, X fastest)
    for (int k = 0; k < 4; ++k)
    { // Z dimension (forward/backward)
        for (int i = 0; i < this->Ny; ++i)
        { // Y dimension (up/down)
            for (int j = 0; j < this->Nx; ++j)
            { // X dimension (left/right)
                // Get interpolated velocity components at cell centers
                float u_val = (j < this->Nx - 1) ? (GetU(i, j) + GetU(i, j + 1)) / 2.0f : 0.0f; // X-component
                float v_val = (i < this->Ny - 1) ? (GetV(i, j) + GetV(i + 1, j)) / 2.0f : 0.0f; // Y-component
                float w_val = 0;

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
double MAC2D::getVatU(int i, int j)
{
    double top = (this->GetV(i, j - 1) + this->GetV(i, j)) / 2.0;
    double botton = (this->GetV(i + 1, j - 1) + this->GetV(i + 1, j)) / 2.0;
    ;

    return (top + botton) / 2.0;
}



double MAC2D::getUatV(int i, int j)
{
    double left = (this->GetU(i, j + 1) + this->GetU(i - 1, j + 1)) / 2.0;
    double right = (this->GetU(i, j) + this->GetU(i - 1, j)) / 2.0;

    return (left + right) / 2.0;
};


void MAC2D::ExportGridOpenFOAM(int iteration)
    {
        std::string levelStr = LevelConfigurationToString(SIMULATION2D.level);
        std::string exportBasePath = "Exports";
        std::string exportPath = exportBasePath + "/" + levelStr + "/" +
                               std::to_string(SIMULATION2D.GRID_SIZE) + "_re" +
                               std::to_string(int(SIMULATION2D.RE)) + "/OpenFOAM/";
        
        // Create the directory structure if it doesn't exist
        std::filesystem::create_directories(exportPath + std::to_string(iteration));
        std::filesystem::create_directories(exportPath + std::to_string(iteration) + "/constant");
        std::filesystem::create_directories(exportPath + std::to_string(iteration) + "/constant/polyMesh");
        
        // Create the .foam file
        std::ofstream foamFile(exportPath + std::to_string(iteration) + "/" + std::to_string(iteration) + ".foam");
        if (!foamFile.is_open()) {
            throw std::runtime_error("Could not create .foam file");
        }
        foamFile << "// OpenFOAM case file\n";
        foamFile.close();
    
    // Step 1: Identify and count all solid cells
    std::vector<std::pair<int, int>> solidCells;
    for (int j = 0; j < this->Ny; ++j) {
        for (int i = 0; i < this->Nx; ++i) {
            if (this->GetSolid(j, i) == SOLID_CELL) {
                solidCells.push_back({i, j});
            }
        }
    }
    
    int numSolidCells = solidCells.size();
    
    // Create a mapping from grid indices to solid cell index
    std::map<std::pair<int, int>, int> solidCellMap;
    for (int idx = 0; idx < numSolidCells; ++idx) {
        solidCellMap[solidCells[idx]] = idx;
    }
    
    // Step 2: Generate points (vertices of solid cells)
    std::ofstream pointsFile(exportPath + std::to_string(iteration) + "/constant/polyMesh/points");
    if (!pointsFile.is_open()) {
        throw std::runtime_error("Could not open points file for writing");
    }
    
    // Calculate total number of points
    int numPoints = 0;
    std::map<std::tuple<int, int>, int> pointMap; // Maps (x,y) to point index
    
    for (const auto& cell : solidCells) {
        int i = cell.first;
        int j = cell.second;
        
        // For each solid cell, we need to define its 4 vertices if they haven't been defined yet
        std::vector<std::tuple<int, int>> vertices = {
            {i, j},         // Bottom-left
            {i+1, j},       // Bottom-right
            {i+1, j+1},     // Top-right
            {i, j+1}        // Top-left
        };
        
        for (const auto& vertex : vertices) {
            if (pointMap.find(vertex) == pointMap.end()) {
                pointMap[vertex] = numPoints++;
            }
        }
    }
    
    // Write points header
    pointsFile << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    pointsFile << "| =========                 |                                                 |\n";
    pointsFile << "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n";
    pointsFile << "|  \\\\    /   O peration     | Version:  v2112                                 |\n";
    pointsFile << "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n";
    pointsFile << "|    \\\\/     M anipulation  |                                                 |\n";
    pointsFile << "\\*---------------------------------------------------------------------------*/\n";
    pointsFile << "FoamFile\n{\n";
    pointsFile << "    version     2.0;\n";
    pointsFile << "    format      ascii;\n";
    pointsFile << "    class       vectorField;\n";
    pointsFile << "    object      points;\n";
    pointsFile << "}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";
    
    pointsFile << numPoints << "\n(\n";
    
    // Sort points by index for consistent output
    std::vector<std::pair<std::tuple<int, int>, int>> sortedPoints;
    for (const auto& entry : pointMap) {
        sortedPoints.push_back({entry.first, entry.second});
    }
    
    std::sort(sortedPoints.begin(), sortedPoints.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& point : sortedPoints) {
        int i = std::get<0>(point.first);
        int j = std::get<1>(point.first);
        // Convert grid indices to physical coordinates
        double x = i * this->dh;
        double y = j * this->dh;
        pointsFile << "(" << x << " " << y << " 0)\n";
    }
    
    pointsFile << ")\n";
    pointsFile << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    pointsFile.close();
    
     // Step 3: Generate faces - Modified to ensure proper boundary face ordering
     std::ofstream facesFile(exportPath + std::to_string(iteration) + "/constant/polyMesh/faces");
     if (!facesFile.is_open()) {
         throw std::runtime_error("Could not open faces file for writing");
     }
 
     std::vector<std::vector<int>> allFaces;
     std::vector<std::pair<int, int>> boundaryFaces; // (faceID, patchID)
     int faceCount = 0;
 
     // First pass: Create all faces and identify boundary faces
     for (const auto& cell : solidCells) {
         int i = cell.first;
         int j = cell.second;
         
         int p0 = pointMap[{i, j}];
         int p1 = pointMap[{i+1, j}];
         int p2 = pointMap[{i+1, j+1}];
         int p3 = pointMap[{i, j+1}];
 
         // Create faces in consistent order: bottom, right, top, left
         std::vector<std::vector<int>> cellFaces = {
             {p0, p1},  // Bottom face (0)
             {p1, p2},  // Right face (1)
             {p2, p3},  // Top face (2)
             {p3, p0}   // Left face (3)
         };
 
         for (int f = 0; f < 4; f++) {
             allFaces.push_back(cellFaces[f]);
             
             // Check if this face is on the boundary
             bool isBoundary = false;
             int patchId = f; // patchId matches face direction
             
             if (f == 0) { // Bottom face
                 isBoundary = (j == 0) || (this->GetSolid(j-1, i) != SOLID_CELL);
             } else if (f == 1) { // Right face
                 isBoundary = (i == this->Nx-1) || (this->GetSolid(j, i+1) != SOLID_CELL);
             } else if (f == 2) { // Top face
                 isBoundary = (j == this->Ny-1) || (this->GetSolid(j+1, i) != SOLID_CELL);
             } else if (f == 3) { // Left face
                 isBoundary = (i == 0) || (this->GetSolid(j, i-1) != SOLID_CELL);
             }
 
             if (isBoundary) {
                 boundaryFaces.push_back({faceCount, patchId});
             }
             faceCount++;
         }
     }
 
     // Write faces file...
     // ... [keep existing faces file writing code] ...
 
     // Step 6: Generate boundary file with proper face ordering
     std::ofstream boundaryFile(exportPath + std::to_string(iteration) + "/constant/polyMesh/boundary");
     if (!boundaryFile.is_open()) {
         throw std::runtime_error("Could not open boundary file for writing");
     }
 
     // Group boundary faces by patch and sort them
     std::map<int, std::vector<int>> patchFaces;
     for (const auto& face : boundaryFaces) {
         patchFaces[face.second].push_back(face.first);
     }
 
     // Sort faces within each patch
     for (auto& patch : patchFaces) {
         std::sort(patch.second.begin(), patch.second.end());
     }
 
     // Calculate start faces for each patch
     int internalFacesCount = allFaces.size() - boundaryFaces.size();
     std::vector<int> startFaces;
     int currentStart = internalFacesCount;
     
     // Patch order: bottom (0), right (1), top (2), left (3)
     for (int patchId = 0; patchId < 4; patchId++) {
         if (patchFaces.count(patchId)) {
             startFaces.push_back(currentStart);
             currentStart += patchFaces[patchId].size();
         }
     }
 
     // Write boundary file...
     boundaryFile << patchFaces.size() << "\n(\n";
     
     std::vector<std::string> patchNames = {"bottom", "right", "top", "left"};
     for (int patchId = 0; patchId < 4; patchId++) {
         if (patchFaces.count(patchId)) {
             boundaryFile << "    " << patchNames[patchId] << "\n";
             boundaryFile << "    {\n";
             boundaryFile << "        type wall;\n";
             boundaryFile << "        nFaces " << patchFaces[patchId].size() << ";\n";
             boundaryFile << "        startFace " << startFaces[patchId] << ";\n";
             boundaryFile << "    }\n";
         }
     }
     boundaryFile << ")\n";
     boundaryFile.close();
 
     // Step 7: Export pressure field with correct face values
     std::ofstream pressureFile(exportPath + std::to_string(iteration) + "/0/p");
     if (!pressureFile.is_open()) {
         throw std::runtime_error("Could not open pressure file for writing");
     }
 
     // ... [keep existing pressure file header] ...
 
     // Boundary field - modified to use correct pressure values
     pressureFile << "boundaryField\n{\n";
     
     for (int patchId = 0; patchId < 4; patchId++) {
         if (patchFaces.count(patchId)) {
             pressureFile << "    " << patchNames[patchId] << "\n";
             pressureFile << "    {\n";
             pressureFile << "        type            calculated;\n";
             pressureFile << "        value           nonuniform List<scalar>\n";
             pressureFile << patchFaces[patchId].size() << "\n(\n";
             
             for (int faceId : patchFaces[patchId]) {
                 int cellId = faceId / 4;
                 int faceDir = faceId % 4;
                 int i = solidCells[cellId].first;
                 int j = solidCells[cellId].second;
                 
                 float pressure = 0.0f;
                 
                 // Get pressure from adjacent cell based on face direction
                 if (faceDir == 0) { // Bottom face
                     pressure = (j > 0) ? this->GetP(j-1, i) : 0.0f;
                 } else if (faceDir == 1) { // Right face
                     pressure = (i+1 < this->Nx) ? this->GetP(j, i+1) : 0.0f;
                 } else if (faceDir == 2) { // Top face
                     pressure = (j+1 < this->Ny) ? this->GetP(j+1, i) : 0.0f;
                 } else if (faceDir == 3) { // Left face
                     pressure = (i > 0) ? this->GetP(j, i-1) : 0.0f;
                 }
                 
                 pressure = std::isfinite(pressure) ? pressure : 0.0f;
                 pressureFile << pressure << "\n";
             }
             
             pressureFile << ");\n";
             pressureFile << "    }\n";
         }
     }
     pressureFile << "}\n";
     pressureFile.close();
 
     std::cout << "OpenFOAM mesh and pressure field exported to " << exportPath << std::endl;
 }