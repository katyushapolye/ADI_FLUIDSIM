FLUID_LAB V0.1

A computational fluid dynamics (CFD) application implementing multiple fluid simulation methods for the Imcompressible Navier-Stokes equations with real-time visualization and real time analysis tools I made during holidays for research and testing different methods and finding their strenghts and weaknesses.
This project provides three distinct fluid simulation approaches, each optimized for different physical scenarios:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b7bc56b-0ada-4539-b3bd-5d0c79ea61f9" alt="Navier-Stokes" width="32%" />
  <img src="https://github.com/user-attachments/assets/65fe26f0-55aa-421a-a05e-ddc790c12b6b" alt="FLIP" width="32%" />
  <img src="https://github.com/user-attachments/assets/f6aef0b2-51be-44f6-b2c5-611333d42781" alt="SPH" width="32%" />
</p>

<p align="center">
  <b>ADI (Eulerian) </b> | <b>FLIP (Hybrid Eulerian-Lagrangian)</b> | <b>SPH (Lagrangian)</b>
</p
  
Validation Test Cases

ADI(3D/2D) Eulerian (Gasses) Solver:
 - Lid-Driven Cavity Flow [1]
 - <img width="1499" height="653" alt="image" src="https://github.com/user-attachments/assets/a5a2324f-73f5-4d81-a3a9-f7b1a6ab6dcd" />

 - Flow Past Obstacle (Wind tunnel with sphere/circle) [2]
 - Backward-Facing Step [3]

   
FLIP(3D/2D) & SPH(2D) (Liquid) Solvers:
 - Dam Break scenario


Currently, the implementation is powered with:
  - OpenMP-accelerated CPU computation with Eigen library for small systems and ADI solutions (With TDMA Algorithm)
  - GPU Acceleration: NVIDIA AMGX library for sparse matrix operations in pressure solving in heavy Poisson Systems.
  - Custom made MAC (Marker-and-Cell) grid layout for optimal memory access patterns with linearized storage and indexing.

Interface:
 - ImGUI-based control panel
 - Real-time plotting with ImPlot (2D graphs)
 - 3D visualization with ImPlot3D
 - ParaView-compatible grid format export (.vtk)
<img width="1616" height="1045" alt="Screenshot from 2025-12-28 14-29-59" src="https://github.com/user-attachments/assets/22ac5ac3-b8d7-4913-8dfe-08aca69a98cb" />



References:

[1] - Ghia, U. K. N. G., Ghia, K. N., & Shin, C. T. "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method" (1982.)

[2] - Schäfer, Michael, et al. "Benchmark computations of laminar flow around a cylinder." (1996.)

[3] - Armaly, Bassem F., et al. "Experimental and theoretical investigation of backward-facing step flow." (1983)\n

