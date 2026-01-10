#include "headers/Solvers/FLIP2D.h"
#include "headers/Solvers/PressureSolver2D.h"
#include "headers/Solvers/DiffusionSolver2D.h"
int FLIP2D::particleCount = 0;
int FLIP2D::maxParticle = 0;
int FLIP2D::particlePerCell = 0.0;
FLIP2D::Particle *FLIP2D::particles = nullptr;
std::vector<std::vector<std::vector<int>>> FLIP2D::SPACE_HASH = std::vector<std::vector<std::vector<int>>>();
MAC FLIP2D::gridAnt = MAC();
double FLIP2D::alpha = 0.0;
Vec2 FLIP2D::queuedAcceleration = Vec2();

void FLIP2D::InitializeFLIP(MAC *grid, double dt, double alpha)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;
    FLIP2D::alpha = alpha;
    maxParticle = SIMULATION.Nx * SIMULATION.Ny * SIMULATION.PARTICLES_PER_CELL;
    FLIP2D::particles = (Particle *)calloc(maxParticle, sizeof(Particle)); // allocate the maximum number possible of particles

    gridAnt.InitializeGrid(SIMULATION.domain,DIMENSION == 3? false:true);
    gridAnt.CopyGrid(*grid);

    // position a test falling block
    double *offset;
    int n = sqrt(SIMULATION.PARTICLES_PER_CELL);
    FLIP2D::particlePerCell = SIMULATION.PARTICLES_PER_CELL;
    offset = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        offset[i] = ((i + 1) * 1.0) / (double)(n + 1);
        // printf("Offset: %f\n",offset[i]);
    }

    for (int i = 0.0 * Ny; i < Ny * 0.4; i++)
    {
        for (int j = 0.0 * Nx; j < Nx * 0.4; j++)
        {
            
            for (int io = 0; io < n; io++)
            {
                for (int jo = 0; jo < n; jo++)
                {
                    Particle p;
                    p.x = j * dh + offset[jo] * dh;
                    p.y = i * dh + offset[io] * dh;
                    p.isActive = true;
                    particles[particleCount] = p;
                    particleCount++;
                }
            }
        }
    }

    ExportParticles(0);
    // initialize the space hash structure
    for (int i = 0; i < Ny; i++)
    {
        SPACE_HASH.push_back(std::vector<std::vector<int>>());
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).push_back(std::vector<int>(SIMULATION.PARTICLES_PER_CELL)); // preallocatigon of some values
        }
    }

    UpdateSpaceHash();

    // falling block for now
    // allocate the space hash
    // create all particles
    //
    free(offset);

    DiffusionSolver2D::InitializeDiffusionSolver(grid);
}

void FLIP2D::FLIP_Momentum(MAC* gridAnt,MAC* gridSol, double time){
        FLIP2D::FLIP_StepBeforeProjection(SIMULATION.GRID_SOL,SIMULATION.dt);
        SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);

        if(SIMULATION.PHYSICAL_DIFFUSION){
            DiffusionSolver2D::SolveDiffusion_Eigen(gridAnt,gridSol);
            
        }
}

void FLIP2D::FLIP_Correction(MAC* grid){
        PressureSolver2D::ProjectPressure(SIMULATION.GRID_SOL); 
        FLIP2D::FLIP_StepAfterProjection(SIMULATION.GRID_SOL,SIMULATION.dt);


}


void FLIP2D::ExportParticles(int IT)
{
    std::string exportPath = "Exports/FLIP/particles_" + std::to_string(IT) + ".csv";
    std::ofstream outFile(exportPath);
    if(!outFile.is_open()){
        std::cout << "FLIP2D EXPORT FAILED!" << std::endl;
    }


    // Write header
    outFile << "x,y,u,v" << std::endl;

    // Write data for each active particle
    for (int i = 0; i < particleCount; i++)
    {
        if (particles[i].isActive)
        {
            outFile << particles[i].x << ","
                    << particles[i].y << ","
                    << particles[i].u << ","
                    << particles[i].v << std::endl;
        }
    }

    outFile.close();
    // std::cout << "Exported " << particleCount << " particles to particles.csv" << std::endl;
}

void FLIP2D::ParticleToGrid(MAC *grid)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;

    // Create data structures for u component
    std::vector<std::vector<double>> uW(Ny, std::vector<double>(Nx + 1, 0.0));
    std::vector<std::vector<double>> uWeights(Ny, std::vector<double>(Nx + 1, 0.0));

    // Create data structures for v component
    std::vector<std::vector<double>> vW(Ny + 1, std::vector<double>(Nx, 0.0));
    std::vector<std::vector<double>> vWeights(Ny + 1, std::vector<double>(Nx, 0.0));

// Main parallel region to avoid thread creation/destruction overhead
#pragma omp parallel
    {
// Process u component (horizontal velocity)
#pragma omp for
        for (int i = 1; i < Ny - 1; i++)
        {
            for (int j = 1; j < Nx; j++)
            {
                std::vector<int> closeParticles;

                // Gather nearby particles
                for (const auto &p : SPACE_HASH[i][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i + 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i + 1][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j - 1])
                    closeParticles.push_back(p);

                // Calculate weights and weighted velocities
                for (const auto &p : closeParticles)
                {
                    double weight = k(particles[p].x - j * dh, particles[p].y - (dh * i + dh / 2.0));
                    uWeights[i][j] += weight;
                    uW[i][j] += particles[p].u * weight;
                }
            }
        }

// Process v component (vertical velocity)
#pragma omp for
        for (int i = 1; i < Ny; i++)
        {
            for (int j = 1; j < Nx - 1; j++)
            {
                std::vector<int> closeParticles;

                // Gather nearby particles
                for (const auto &p : SPACE_HASH[i][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j + 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j + 1])
                    closeParticles.push_back(p);

                // Calculate weights and weighted velocities
                for (const auto &p : closeParticles)
                {
                    double weight = k(particles[p].x - (dh * j + dh / 2.0), particles[p].y - (dh * i));
                    vWeights[i][j] += weight;
                    vW[i][j] += particles[p].v * weight;
                }
            }
        }

// Normalize and set u velocity values
#pragma omp for
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx + 1; j++)
            {
                // Set values below tolerance to zero
                if (std::abs(uW[i][j]) < 1e-12)
                    uW[i][j] = 0.0;
                if (std::abs(uWeights[i][j]) < 1e-12)
                    uWeights[i][j] = 0.0;

                // Divide velocity by weight where weight is not zero
                if (uWeights[i][j] != 0.0)
                {
                    grid->SetU(i, j, uW[i][j] / uWeights[i][j]);
                }
                else
                {
                    grid->SetU(i, j, 0.0);
                }
            }
        }

// Normalize and set v velocity values
#pragma omp for
        for (int i = 0; i < Ny + 1; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                // Set values below tolerance to zero
                if (std::abs(vW[i][j]) < 1e-12)
                    vW[i][j] = 0.0;
                if (std::abs(vWeights[i][j]) < 1e-12)
                    vWeights[i][j] = 0.0;

                // Divide velocity by weight where weight is not zero
                if (vWeights[i][j] != 0.0)
                {
                    grid->SetV(i, j, vW[i][j] / vWeights[i][j]);
                }
                else
                {
                    grid->SetV(i, j, 0.0);
                }
            }
        }
    } // End of parallel region
}

void FLIP2D::GridToParticle(MAC *grid)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;

    // Calculate the velocity difference LATER
    std::vector<std::vector<double>> du(Ny, std::vector<double>(Nx + 1, 0.0));
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            du[i][j] = grid->GetU(i, j) - gridAnt.GetU(i, j);
        }
    }

    // u velocity component
    std::vector<double> uW(particleCount, 0.0);  // weight times the velocity
    std::vector<double> duW(particleCount, 0.0); // weight times the velocity difference
    std::vector<double> wu(particleCount, 0.0);  // weight for each particle
#pragma omp for
    for (int p = 0; p < particleCount; p++)
    {
        // we check only our nearby faces, for u, it is the to
        int pJ = particles[p].x / dh; // by the logic we are using, this should never segfault, but if it does, it is this 100%
        int pI = (particles[p].y + (dh / 2.0)) / dh;
        int pIend = pI + 2 > (Ny) ? pI + 1 : pI + 2;
        int pJend = pJ + 2 > (Nx + 1) ? pJ + 1 : pJ + 2;

        for (int i = pI - 1; i < pIend; i++)
        {
            for (int j = pJ - 1; j < pJend; j++)
            {
                double weight = k(particles[p].x - j * dh, particles[p].y - (i * dh + dh / 2.0));
                wu[p] += weight;
                uW[p] += grid->GetU(i, j) * weight;
                duW[p] += du[i][j] * weight;
            }
        }

        // Set values below tolerance to zero
        if (std::abs(uW[p]) < 1e-12)
            uW[p] = 0.0;
        if (std::abs(wu[p]) < 1e-12)
            wu[p] = 0.0;

        // Update particle velocity using FLIP2D/PIC blend
        if (wu[p] != 0.0)
        {
            particles[p].u = (uW[p] / wu[p]) * (1.0 - alpha) + (particles[p].u + (duW[p] / wu[p])) * alpha;
        }
    }

    // Calculate the velocity difference for v component
    std::vector<std::vector<double>> dv(Ny + 1, std::vector<double>(Nx, 0.0));
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            dv[i][j] = grid->GetV(i, j) - gridAnt.GetV(i, j);
        }
    }

    // v velocity component
    std::vector<double> vW(particleCount, 0.0);  // weight times the velocity
    std::vector<double> dvW(particleCount, 0.0); // weight times the velocity difference
    std::vector<double> wv(particleCount, 0.0);  // weight for each particle

#pragma omp for
    for (int p = 0; p < particleCount; p++)
    {

        int pJ = (particles[p].x + (dh / 2.0)) / dh; // by the logic we are using, this should never segfault, but if it does, it is this 100%
        int pI = (particles[p].y) / dh;

        int pIend = pI + 2 > (Ny + 1) ? pI + 1 : pI + 2;
        int pJend = pJ + 2 > (Nx) ? pJ + 1 : pJ + 2;

        for (int i = pI - 1; i < pIend; i++)
        {
            for (int j = pJ - 1; j < pJend; j++)
            {
                double weight = k(particles[p].x - (j * dh + dh / 2.0), particles[p].y - (i * dh));
                wv[p] += weight;
                vW[p] += grid->GetV(i, j) * weight;
                dvW[p] += dv[i][j] * weight;
            }
        }

        // Set values below tolerance to zero
        if (std::abs(vW[p]) < 1e-12)
            vW[p] = 0.0;
        if (std::abs(wv[p]) < 1e-12)
            wv[p] = 0.0;

        // Update particle velocity using FLIP2D/PIC blend
        if (wv[p] != 0.0)
        {
            particles[p].v = (vW[p] / wv[p]) * (1.0 - alpha) + (particles[p].v + (dvW[p] / wv[p])) * alpha;
        }
    }
}




// change the numerical method here
void FLIP2D::UpdateParticles(double dt)
{
    for (int p = 0; p < particleCount; p++)
    {


        particles[p].x += particles[p].u * dt;
        particles[p].y += particles[p].v * dt;
        // small boudnary chheck for sanitty!
        if (particles[p].x >= SIMULATION.domain.xf - SIMULATION.dh)
        {
            particles[p].x = SIMULATION.domain.xf - (SIMULATION.dh * 1.0001);
            particles[p].u = 0.0;
            particles[p].v = 0.0;
        }

        if (particles[p].x <= SIMULATION.domain.x0 + SIMULATION.dh)
        {
            particles[p].x = SIMULATION.domain.x0 + (SIMULATION.dh * 1.0001);
            particles[p].u = 0.0;
            particles[p].v = 0.0;
        }

        if (particles[p].y >= SIMULATION.domain.yf - SIMULATION.dh)
        {
            particles[p].y = SIMULATION.domain.yf - (SIMULATION.dh * 1.0001);
            particles[p].v = 0.0;
            particles[p].u = 0.0;
        }

        if (particles[p].y <= SIMULATION.domain.y0 + SIMULATION.dh)
        {
            particles[p].y = SIMULATION.domain.y0 + (SIMULATION.dh * 1.0001);
            particles[p].v = 0.0;
            particles[p].u = 0.0;
        }
    }
}

void FLIP2D::UpdateSpaceHash()
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    // cleaning up
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).at(j).clear();
        }
    }

    int i, j;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = particles[p].y / dh;
        j = particles[p].x / dh; // this implicit downmcast takes care of thhe floor

        SPACE_HASH.at(i).at(j).push_back(p);

        // pushes to the list
    }
}
void FLIP2D::UpdateFluidCells(MAC *grid)
{
    grid->ResetFluidCells();

    int i, j;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = particles[p].y / dh;
        j = particles[p].x / dh; // this implicit downmcast takes care of thhe floor
        i = i > (SIMULATION.Ny - 1) ? SIMULATION.Ny - 1 : i;
        j = j > (SIMULATION.Nx - 1) ? SIMULATION.Nx - 1 : j;

        if (grid->GetSolid(i, j) == EMPTY_CELL)
        {
            grid->SetSolid(i, j, FLUID_CELL);
        }

        // check all neightboors to no have problems
    }
    // check for holes

    for (int i = 1; i < grid->Ny - 1; i++)
    {
        for (int j = 1; j < grid->Nx - 1; j++)
        {

            int nonEmptyNeighbors = 0;
            if (grid->GetSolid(i, j + 1) != EMPTY_CELL) nonEmptyNeighbors++;
            if (grid->GetSolid(i, j - 1) != EMPTY_CELL) nonEmptyNeighbors++;
            if (grid->GetSolid(i + 1, j) != EMPTY_CELL) nonEmptyNeighbors++;
            if (grid->GetSolid(i - 1, j) != EMPTY_CELL) nonEmptyNeighbors++;
            
            if (grid->GetSolid(i, j) == EMPTY_CELL && nonEmptyNeighbors >= 3)
            {
                grid->SetSolid(i, j, FLUID_CELL);
            }

            // update the grid solid mask for the pressure step, rewrite a bit of thhe pressure so that it remakes the pressure matrix each step
        }
    }
}

/*

double FLIP2D::h(double r){
    if(r >= 0.0 && r <= 1.0){
        return 1.0-r;
    }
    if(r <= 0.0 && r>= -1.0){
        return 1.0+r;
    }
    else{
        return 0;
    }
}
double FLIP2D::k(double x,double y){
    return h(x/SIMULATION.dh)*h(y/SIMULATION.dh);
}
*/

double FLIP2D::h(double r)
{
    r = std::abs(r); // Take absolute value for symmetry

    if (r >= 0.0 && r <= 0.5)
    {
        return 0.75 - r * r;
    }
    else if (r > 0.5 && r <= 1.5)
    {
        return 0.5 * pow(1.5 - r, 2);
    }
    else
    {
        return 0.0;
    }
}

// 2D kernel using tensor product of 1D kernels
double FLIP2D::k(double x, double y)
{
    return h(x / SIMULATION.dh) * h(y / SIMULATION.dh);
}

void FLIP2D::FLIP_StepBeforeProjection(MAC *grid, double dt)
{
    // impose boundary
    // change the external force later
    double start = omp_get_wtime();
    //
    grid->SetBorder(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D, 0); // change here later

    UpdateParticles(dt);
    UpdateFluidCells(grid);
    UpdateSpaceHash();

    ParticleToGrid(grid); // copy the grid here for flip

    grid->SetBorder(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D, 0);

    gridAnt.CopyGrid(*grid); // saving the past grid for flip

    Vec2 g;
    
    g.u = SIMULATION.f.u;// * sin((SIMULATION.rotation*M_PI)/180.0);
    g.v = SIMULATION.f.v;// * cos((SIMULATION.rotation*M_PI)/180.0);

    // External program forces
    g.u = g.u + queuedAcceleration.u;//* cos((SIMULATION.rotation*M_PI)/180.0);
    g.v = g.v + queuedAcceleration.v;//* cos((SIMULATION.rotation*M_PI)/180.0);

    grid->AddAcceleration(g, dt);

    double end = omp_get_wtime();
    SIMULATION.lastParticleTime = end-start;
}
void FLIP2D::FLIP_StepAfterProjection(MAC *grid, double dt)
{
    double start = omp_get_wtime();
    //grid->SetBorder(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D, 0); // change here later
    GridToParticle(grid);

    double end = omp_get_wtime();
    SIMULATION.lastParticleTime += end-start;
}

void FLIP2D::QueueAcceleration(Vec2 a)
{
    queuedAcceleration = a;
}

double FLIP2D::GetTotalKinecticEnergy(){
    //this is a rough approximation
    //since we have a cell size and particles per cell, we assume that each particle has a mass
    //by the cell size and density we have the mass (all in meters) 1000Kg per meter cubed, lets assume everything has 1 meter depth
    //so our particles mass is pW = 1000*dh*dh*1 / Particle nuum
    double m = 180.0 / FLIP2D::particleCount; //cghange here
    double v =0;
    double e = 0.0;
    for (int p = 0; p < particleCount; p++)
    {
        v = sqrt(particles[p].u * particles[p].u + particles[p].v * particles[p].v);
        e+= 0.5*(m*v*v);

    }

    return e;





}

double FLIP2D::GetTotalPotentialEnergy(){
    //this is a rough approximation
    //since we have a cell size and particles per cell, we assume that each particle has a mass
    //by the cell size and density we have the mass (all in meters) 1000Kg per meter cubed, lets assume everything has 1 meter depth
    //so our particles mass is pW = 1000*dh*dh*1 / Particle nuum
    //double m = (SIMULATION.RHO * SIMULATION.dh *SIMULATION.dh * 1.0) / FLIP2D::particlePerCell;
    double m = 180.0 / FLIP2D::particleCount; 
    double h = 0.0;
    double g = SIMULATION.f.v;
    double e = 0.0;
    for (int p = 0; p < particleCount; p++)
    {
        h = particles[p].y;
        e+= abs(g*h*m);

    }

    return e;





}
