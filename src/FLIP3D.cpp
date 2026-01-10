#include "headers/Solvers/FLIP3D.h"
#include "headers/Solvers/PressureSolver3D.h"
#include "headers/Solvers/DiffusionSolver3D.h"

int FLIP3D::particleCount = 0;
int FLIP3D::maxParticle = 0;
int FLIP3D::particlePerCell = 0.0;
FLIP3D::Particle *FLIP3D::particles = nullptr;
std::vector<std::vector<std::vector<std::vector<int>>>> FLIP3D::SPACE_HASH = std::vector<std::vector<std::vector<std::vector<int>>>>();
MAC FLIP3D::gridAnt = MAC();
double FLIP3D::alpha = 0.0;
Vec3 FLIP3D::queuedAcceleration = Vec3();

void FLIP3D::InitializeFLIP(MAC *grid, double dt, double alpha)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;
    double dh = SIMULATION.dh;
    FLIP3D::alpha = alpha;
    maxParticle = SIMULATION.Nx * SIMULATION.Ny * SIMULATION.Nz * SIMULATION.PARTICLES_PER_CELL;
    FLIP3D::particles = (Particle *)calloc(maxParticle, sizeof(Particle));

    gridAnt.InitializeGrid(SIMULATION.domain, false);
    gridAnt.CopyGrid(*grid);

    // Position particles in initial volume
    double *offset;
    int n = cbrt(SIMULATION.PARTICLES_PER_CELL);
    FLIP3D::particlePerCell = SIMULATION.PARTICLES_PER_CELL; //handle
    offset = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        offset[i] = ((i + 1) * 1.0) / (double)(n + 1);
    }

    for (int i = 0.0 * Ny; i < Ny * 0.5; i++)
    {
        for (int j = 0.0 * Nx; j < Nx * 0.5; j++)
        {
            for (int k = 0.0 * Nz; k < Nz * 1.0; k++)
            {
                for (int io = 0; io < n; io++)
                {
                    for (int jo = 0; jo < n; jo++)
                    {
                        for (int ko = 0; ko < n; ko++)
                        {
                            Particle p;
                            p.x = j * dh + offset[jo] * dh;
                            p.y = i * dh + offset[io] * dh;
                            p.z = k * dh + offset[ko] * dh;
                            p.isActive = true;
                            particles[particleCount] = p;
                            particleCount++;
                        }
                    }
                }
            }
        }
    }

    ExportParticles(0);

    
    // Initialize 3D space hash structure
    for (int i = 0; i < Ny; i++)
    {
        SPACE_HASH.push_back(std::vector<std::vector<std::vector<int>>>());
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).push_back(std::vector<std::vector<int>>());
            for (int k = 0; k < Nz; k++)
            {
                SPACE_HASH.at(i).at(j).push_back(std::vector<int>(SIMULATION.PARTICLES_PER_CELL));
            }
        }
    }

    UpdateSpaceHash();
    free(offset);
    DiffusionSolver3D::InitializeDiffusionSolver(grid);
}

void FLIP3D::FLIP_Momentum(MAC* gridAnt, MAC* gridSol, double time)
{
    FLIP3D::FLIP_StepBeforeProjection(SIMULATION.GRID_SOL, SIMULATION.dt);
    SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);
    if(SIMULATION.PHYSICAL_DIFFUSION){
            SIMULATION.GRID_SOL->SetGrid(ZERO,ZERO_SCALAR,0.0);
            DiffusionSolver3D::SolveDiffusion_Eigen(SIMULATION.GRID_ANT,SIMULATION.GRID_SOL);
            //ADI2D::SolveADIDiffusionStep(SIMULATION.GRID_ANT,SIMULATION.GRID_SOL,0);
    }
}

void FLIP3D::FLIP_Correction(MAC* grid)
{
    PressureSolver3D::ProjectPressure(SIMULATION.GRID_SOL); //
    FLIP3D::FLIP_StepAfterProjection(SIMULATION.GRID_SOL, SIMULATION.dt);
}

void FLIP3D::ExportParticles(int IT)
{
    std::string exportPath = "Exports/FLIP/particles_" + std::to_string(IT) + ".csv";
    std::ofstream outFile(exportPath);

    if(!outFile.is_open()){
        std::cout << "FLIP EXPORT FAILED!" << std::endl;
        return;
    }

    outFile << "x,y,z,u,v,w" << std::endl;

    for (int i = 0; i < particleCount; i++)
    {
        if (particles[i].isActive)
        {
            outFile << particles[i].x << ","
                    << particles[i].y << ","
                    << particles[i].z << ","
                    << particles[i].u << ","
                    << particles[i].v << ","
                    << particles[i].w << std::endl;
        }
    }

    outFile.close();
}

void FLIP3D::ParticleToGrid(MAC *grid)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;
    double dh = SIMULATION.dh;

    // Data structures for u component (x-velocity on x-faces)
    std::vector<std::vector<std::vector<double>>> uW(Ny, std::vector<std::vector<double>>(Nx + 1, std::vector<double>(Nz, 0.0)));
    std::vector<std::vector<std::vector<double>>> uWeights(Ny, std::vector<std::vector<double>>(Nx + 1, std::vector<double>(Nz, 0.0)));

    // Data structures for v component (y-velocity on y-faces)
    std::vector<std::vector<std::vector<double>>> vW(Ny + 1, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz, 0.0)));
    std::vector<std::vector<std::vector<double>>> vWeights(Ny + 1, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz, 0.0)));

    // Data structures for w component (z-velocity on z-faces)
    std::vector<std::vector<std::vector<double>>> wW(Ny, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz + 1, 0.0)));
    std::vector<std::vector<std::vector<double>>> wWeights(Ny, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz + 1, 0.0)));

#pragma omp parallel
    {
// Process u component (x-velocity)
#pragma omp for
        for (int i = 1; i < Ny - 1; i++)
        {
            for (int j = 1; j < Nx; j++)
            {
                for (int k = 1; k < Nz - 1; k++)
                {
                    std::vector<int> closeParticles;

                    // Gather particles from nearby cells
                    for (int di = -1; di <= 1; di++)
                    {
                        for (int dj = -1; dj <= 0; dj++)
                        {
                            for (int dk = -1; dk <= 1; dk++)
                            {
                                for (const auto &p : SPACE_HASH[i + di][j + dj][k + dk])
                                    closeParticles.push_back(p);
                            }
                        }
                    }

                    // Calculate weights and weighted velocities
                    for (const auto &p : closeParticles)
                    {
                        double weight = FLIP3D::k(particles[p].x - j * dh, 
                                        particles[p].y - (i * dh + dh / 2.0),
                                        particles[p].z - (k * dh + dh / 2.0));
                        uWeights[i][j][k] += weight;
                        uW[i][j][k] += particles[p].u * weight;
                    }
                }
            }
        }

// Process v component (y-velocity)
#pragma omp for
        for (int i = 1; i < Ny; i++)
        {
            for (int j = 1; j < Nx - 1; j++)
            {
                for (int k = 1; k < Nz - 1; k++)
                {
                    std::vector<int> closeParticles;

                    // Gather particles from nearby cells
                    for (int di = -1; di <= 0; di++)
                    {
                        for (int dj = -1; dj <= 1; dj++)
                        {
                            for (int dk = -1; dk <= 1; dk++)
                            {
                                for (const auto &p : SPACE_HASH[i + di][j + dj][k + dk])
                                    closeParticles.push_back(p);
                            }
                        }
                    }

                    // Calculate weights and weighted velocities
                    for (const auto &p : closeParticles)
                    {
                        double weight = FLIP3D::k(particles[p].x - (j * dh + dh / 2.0),
                                        particles[p].y - i * dh,
                                        particles[p].z - (k * dh + dh / 2.0));
                        vWeights[i][j][k] += weight;
                        vW[i][j][k] += particles[p].v * weight;
                    }
                }
            }
        }

// Process w component (z-velocity)
#pragma omp for
        for (int i = 1; i < Ny - 1; i++)
        {
            for (int j = 1; j < Nx - 1; j++)
            {
                for (int k = 1; k < Nz; k++)
                {
                    std::vector<int> closeParticles;

                    // Gather particles from nearby cells
                    for (int di = -1; di <= 1; di++)
                    {
                        for (int dj = -1; dj <= 1; dj++)
                        {
                            for (int dk = -1; dk <= 0; dk++)
                            {
                                for (const auto &p : SPACE_HASH[i + di][j + dj][k + dk])
                                    closeParticles.push_back(p);
                            }
                        }
                    }

                    // Calculate weights and weighted velocities
                    for (const auto &p : closeParticles)
                    {
                        double weight = FLIP3D::k(particles[p].x - (j * dh + dh / 2.0),
                                        particles[p].y - (i * dh + dh / 2.0),
                                        particles[p].z - k * dh);
                        wWeights[i][j][k] += weight;
                        wW[i][j][k] += particles[p].w * weight;
                    }
                }
            }
        }

// Normalize and set u velocity values
#pragma omp for
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx + 1; j++)
            {
                for (int k = 0; k < Nz; k++)
                {
                    if (std::abs(uW[i][j][k]) < 1e-12)
                        uW[i][j][k] = 0.0;
                    if (std::abs(uWeights[i][j][k]) < 1e-12)
                        uWeights[i][j][k] = 0.0;

                    if (uWeights[i][j][k] != 0.0)
                    {
                        grid->SetU(i, j, k, uW[i][j][k] / uWeights[i][j][k]);
                    }
                    else
                    {
                        grid->SetU(i, j, k, 0.0);
                    }
                }
            }
        }

// Normalize and set v velocity values
#pragma omp for
        for (int i = 0; i < Ny + 1; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                for (int k = 0; k < Nz; k++)
                {
                    if (std::abs(vW[i][j][k]) < 1e-12)
                        vW[i][j][k] = 0.0;
                    if (std::abs(vWeights[i][j][k]) < 1e-12)
                        vWeights[i][j][k] = 0.0;

                    if (vWeights[i][j][k] != 0.0)
                    {
                        grid->SetV(i, j, k, vW[i][j][k] / vWeights[i][j][k]);
                    }
                    else
                    {
                        grid->SetV(i, j, k, 0.0);
                    }
                }
            }
        }

// Normalize and set w velocity values
#pragma omp for
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                for (int k = 0; k < Nz + 1; k++)
                {
                    if (std::abs(wW[i][j][k]) < 1e-12)
                        wW[i][j][k] = 0.0;
                    if (std::abs(wWeights[i][j][k]) < 1e-12)
                        wWeights[i][j][k] = 0.0;

                    if (wWeights[i][j][k] != 0.0)
                    {
                        grid->SetW(i, j, k, wW[i][j][k] / wWeights[i][j][k]);
                    }
                    else
                    {
                        grid->SetW(i, j, k, 0.0);
                    }
                }
            }
        }
    }
}

void FLIP3D::GridToParticle(MAC *grid)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;
    double dh = SIMULATION.dh;

    // Calculate velocity differences for u component
    std::vector<std::vector<std::vector<double>>> du(Ny, std::vector<std::vector<double>>(Nx + 1, std::vector<double>(Nz, 0.0)));
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                du[i][j][k] = grid->GetU(i, j, k) - gridAnt.GetU(i, j, k);
            }
        }
    }

    // u velocity component
    std::vector<double> uW(particleCount, 0.0);
    std::vector<double> duW(particleCount, 0.0);
    std::vector<double> wu(particleCount, 0.0);

#pragma omp for
    for (int p = 0; p < particleCount; p++)
    {
        int pJ = particles[p].x / dh;
        int pI = (particles[p].y + (dh / 2.0)) / dh;
        int pK = (particles[p].z + (dh / 2.0)) / dh;
        
        int pIend = pI + 2 > Ny ? pI + 1 : pI + 2;
        int pJend = pJ + 2 > (Nx + 1) ? pJ + 1 : pJ + 2;
        int pKend = pK + 2 > Nz ? pK + 1 : pK + 2;

        for (int i = pI - 1; i < pIend; i++)
        {
            for (int j = pJ - 1; j < pJend; j++)
            {
                for (int k = pK - 1; k < pKend; k++)
                {
                    double weight = FLIP3D::k(particles[p].x - j * dh, 
                                    particles[p].y - (i * dh + dh / 2.0),
                                    particles[p].z - (k * dh + dh / 2.0));
                    wu[p] += weight;
                    uW[p] += grid->GetU(i, j, k) * weight;
                    duW[p] += du[i][j][k] * weight;
                }
            }
        }

        if (std::abs(uW[p]) < 1e-12)
            uW[p] = 0.0;
        if (std::abs(wu[p]) < 1e-12)
            wu[p] = 0.0;

        if (wu[p] != 0.0)
        {
            particles[p].u = (uW[p] / wu[p]) * (1.0 - alpha) + (particles[p].u + (duW[p] / wu[p])) * alpha;
        }
    }

    // Calculate velocity differences for v component
    std::vector<std::vector<std::vector<double>>> dv(Ny + 1, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz, 0.0)));
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                dv[i][j][k] = grid->GetV(i, j, k) - gridAnt.GetV(i, j, k);
            }
        }
    }

    // v velocity component
    std::vector<double> vW(particleCount, 0.0);
    std::vector<double> dvW(particleCount, 0.0);
    std::vector<double> wv(particleCount, 0.0);

#pragma omp for
    for (int p = 0; p < particleCount; p++)
    {
        int pJ = (particles[p].x + (dh / 2.0)) / dh;
        int pI = particles[p].y / dh;
        int pK = (particles[p].z + (dh / 2.0)) / dh;

        int pIend = pI + 2 > (Ny + 1) ? pI + 1 : pI + 2;
        int pJend = pJ + 2 > Nx ? pJ + 1 : pJ + 2;
        int pKend = pK + 2 > Nz ? pK + 1 : pK + 2;

        for (int i = pI - 1; i < pIend; i++)
        {
            for (int j = pJ - 1; j < pJend; j++)
            {
                for (int k = pK - 1; k < pKend; k++)
                {
                    double weight = FLIP3D::k(particles[p].x - (j * dh + dh / 2.0),
                                    particles[p].y - i * dh,
                                    particles[p].z - (k * dh + dh / 2.0));
                    wv[p] += weight;
                    vW[p] += grid->GetV(i, j, k) * weight;
                    dvW[p] += dv[i][j][k] * weight;
                }
            }
        }

        if (std::abs(vW[p]) < 1e-12)
            vW[p] = 0.0;
        if (std::abs(wv[p]) < 1e-12)
            wv[p] = 0.0;

        if (wv[p] != 0.0)
        {
            particles[p].v = (vW[p] / wv[p]) * (1.0 - alpha) + (particles[p].v + (dvW[p] / wv[p])) * alpha;
        }
    }

    // Calculate velocity differences for w component
    std::vector<std::vector<std::vector<double>>> dw(Ny, std::vector<std::vector<double>>(Nx, std::vector<double>(Nz + 1, 0.0)));
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz + 1; k++)
            {
                dw[i][j][k] = grid->GetW(i, j, k) - gridAnt.GetW(i, j, k);
            }
        }
    }

    // w velocity component
    std::vector<double> wW(particleCount, 0.0);
    std::vector<double> dwW(particleCount, 0.0);
    std::vector<double> ww(particleCount, 0.0);

#pragma omp for
    for (int p = 0; p < particleCount; p++)
    {
        int pJ = (particles[p].x + (dh / 2.0)) / dh;
        int pI = (particles[p].y + (dh / 2.0)) / dh;
        int pK = particles[p].z / dh;

        int pIend = pI + 2 > Ny ? pI + 1 : pI + 2;
        int pJend = pJ + 2 > Nx ? pJ + 1 : pJ + 2;
        int pKend = pK + 2 > (Nz + 1) ? pK + 1 : pK + 2;

        for (int i = pI - 1; i < pIend; i++)
        {
            for (int j = pJ - 1; j < pJend; j++)
            {
                for (int k = pK - 1; k < pKend; k++)
                {
                    double weight = FLIP3D::k(particles[p].x - (j * dh + dh / 2.0),
                                    particles[p].y - (i * dh + dh / 2.0),
                                    particles[p].z - k * dh);
                    ww[p] += weight;
                    wW[p] += grid->GetW(i, j, k) * weight;
                    dwW[p] += dw[i][j][k] * weight;
                }
            }
        }

        if (std::abs(wW[p]) < 1e-12)
            wW[p] = 0.0;
        if (std::abs(ww[p]) < 1e-12)
            ww[p] = 0.0;

        if (ww[p] != 0.0)
        {
            particles[p].w = (wW[p] / ww[p]) * (1.0 - alpha) + (particles[p].w + (dwW[p] / ww[p])) * alpha;
        }
    }
}

void FLIP3D::UpdateParticles(double dt)
{
    for (int p = 0; p < particleCount; p++)
    {
        particles[p].x += particles[p].u * dt;
        particles[p].y += particles[p].v * dt;
        particles[p].z += particles[p].w * dt;

        // Boundary checks for x
        if (particles[p].x >= SIMULATION.domain.xf - SIMULATION.dh)
        {
            particles[p].x = SIMULATION.domain.xf - (SIMULATION.dh * 1.0001);
            particles[p].u = 0.0;
            particles[p].v = 0.0;
            particles[p].w = 0.0;
        }
        if (particles[p].x <= SIMULATION.domain.x0 + SIMULATION.dh)
        {
            particles[p].x = SIMULATION.domain.x0 + (SIMULATION.dh * 1.0001);
            particles[p].u = 0.0;
            particles[p].v = 0.0;
            particles[p].w = 0.0;
        }

        // Boundary checks for y
        if (particles[p].y >= SIMULATION.domain.yf - SIMULATION.dh)
        {
            particles[p].y = SIMULATION.domain.yf - (SIMULATION.dh * 1.0001);
            particles[p].v = 0.0;
            particles[p].u = 0.0;
            particles[p].w = 0.0;
        }
        if (particles[p].y <= SIMULATION.domain.y0 + SIMULATION.dh)
        {
            particles[p].y = SIMULATION.domain.y0 + (SIMULATION.dh * 1.0001);
            particles[p].v = 0.0;
            particles[p].u = 0.0;
            particles[p].w = 0.0;
        }

        // Boundary checks for z
        if (particles[p].z >= SIMULATION.domain.zf - SIMULATION.dh)
        {
            particles[p].z = SIMULATION.domain.zf - (SIMULATION.dh * 1.0001);
            particles[p].w = 0.0;
            particles[p].u = 0.0;
            particles[p].v = 0.0;
        }
        if (particles[p].z <= SIMULATION.domain.z0 + SIMULATION.dh)
        {
            particles[p].z = SIMULATION.domain.z0 + (SIMULATION.dh * 1.0001);
            particles[p].w = 0.0;
            particles[p].u = 0.0;
            particles[p].v = 0.0;
        }
    }
}

void FLIP3D::UpdateSpaceHash()
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;
    
    // Clear existing hash
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                SPACE_HASH.at(i).at(j).at(k).clear();
            }
        }
    }

    int i, j, k;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = particles[p].y / dh;
        j = particles[p].x / dh;
        k = particles[p].z / dh;

        SPACE_HASH.at(i).at(j).at(k).push_back(p);
    }
}

void FLIP3D::UpdateFluidCells(MAC *grid)
{
    grid->ResetFluidCells();

    int i, j, k;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = particles[p].y / dh;
        j = particles[p].x / dh;
        k = particles[p].z / dh;
        
        i = i > (SIMULATION.Ny - 1) ? SIMULATION.Ny - 1 : i;
        j = j > (SIMULATION.Nx - 1) ? SIMULATION.Nx - 1 : j;
        k = k > (SIMULATION.Nz - 1) ? SIMULATION.Nz - 1 : k;

        if (grid->GetSolid(i, j, k) == EMPTY_CELL)
        {
            grid->SetSolid(i, j, k, FLUID_CELL);
        }
    }

    // Check for holes and fill them
    for (int i = 1; i < grid->Ny - 1; i++)
    {
        for (int j = 1; j < grid->Nx - 1; j++)
        {
            for (int k = 1; k < grid->Nz - 1; k++)
            {
                int nonEmptyNeighbors = 0;
                if (grid->GetSolid(i, j + 1, k) != EMPTY_CELL) nonEmptyNeighbors++;
                if (grid->GetSolid(i, j - 1, k) != EMPTY_CELL) nonEmptyNeighbors++;
                if (grid->GetSolid(i + 1, j, k) != EMPTY_CELL) nonEmptyNeighbors++;
                if (grid->GetSolid(i - 1, j, k) != EMPTY_CELL) nonEmptyNeighbors++;
                if (grid->GetSolid(i, j, k + 1) != EMPTY_CELL) nonEmptyNeighbors++;
                if (grid->GetSolid(i, j, k - 1) != EMPTY_CELL) nonEmptyNeighbors++;
                
                if (grid->GetSolid(i, j, k) == EMPTY_CELL && nonEmptyNeighbors >= 5)
                {
                    grid->SetSolid(i, j, k, FLUID_CELL);
                }
            }
        }
    }
}

double FLIP3D::h(double r)
{
    r = std::abs(r);

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

// 3D kernel using tensor product of 1D kernels
double FLIP3D::k(double x, double y, double z)
{
    return h(x / SIMULATION.dh) * h(y / SIMULATION.dh) * h(z / SIMULATION.dh);
}

void FLIP3D::FLIP_StepBeforeProjection(MAC *grid, double dt)
{      
    double start = omp_get_wtime();
    grid->SetBorder(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction, 0);

    UpdateParticles(dt);

    UpdateFluidCells(grid);
    UpdateSpaceHash();

    ParticleToGrid(grid);

    grid->SetBorder(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction, 0);

    gridAnt.CopyGrid(*grid);

    Vec3 g;
    g.u = SIMULATION.f.u;
    g.v = SIMULATION.f.v;
    g.w = SIMULATION.f.w;

    // External program forces
    g.u = g.u + queuedAcceleration.u;
    g.v = g.v + queuedAcceleration.v;
    g.w = g.w + queuedAcceleration.w;

    grid->AddAcceleration(g, dt);
    double end = omp_get_wtime();

    SIMULATION.lastParticleUpdateTime = end - start;
}

void FLIP3D::FLIP_StepAfterProjection(MAC *grid, double dt)

{   double start = omp_get_wtime();
    GridToParticle(grid);
    double end = omp_get_wtime();

    SIMULATION.lastParticleUpdateTime += end - start;
}

void FLIP3D::QueueAcceleration(Vec3 a)
{
    queuedAcceleration = a;
}

double FLIP3D::GetTotalKineticEnergy()
{
    double m = 180.0 / FLIP3D::particleCount;
    double v = 0;
    double e = 0.0;
    
    for (int p = 0; p < particleCount; p++)
    {
        v = sqrt(particles[p].u * particles[p].u + 
                 particles[p].v * particles[p].v + 
                 particles[p].w * particles[p].w);
        e += 0.5 * (m * v * v);
    }

    return e;
}

double FLIP3D::GetTotalPotentialEnergy()
{
    double m = 180.0 / FLIP3D::particleCount;
    double h = 0.0;
    double g = SIMULATION.f.v;
    double e = 0.0;
    
    for (int p = 0; p < particleCount; p++)
    {
        h = particles[p].y;
        e += abs(g * h * m);
    }

    return e;
}