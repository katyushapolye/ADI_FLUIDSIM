#ifndef EXACT_H
#define EXACT_H
#include "Utils.h"
#include "math.h"
#include "Definitions.h"




/*//////////////////////////////////////////////
//////Cavity (Check the Threashold) ////////////
*///////////////////////////////////////////////

inline Vec3 LID_CAVITY_FLOW(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    if(y >= 1.0 - (SIMULATION.dh*0.55)){
        r.u = 1.0;
    }


    return r;

}

inline double LID_CAVITY_FLOW_PRESSURE(double x, double y,double z,double t){

    return 0.0;

}


/*//////////////////////////////////////////////
///////////Backwards Facing Step ///////////////
*///////////////////////////////////////////////

inline Vec3 BACKWARDS_FACING_STEP(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    double dh = SIMULATION.dh;
    double S = 0.49;
    double U0 = 2.0;
    //on the top of the step

    if(x <= SIMULATION.dh  && (z<=1-SIMULATION.dh/2.0) && z>=SIMULATION.dh/2.0 && y <= (1.0 - SIMULATION.dh/2.0) && y >= SIMULATION.dh/2 && y > S){
        r.u = 1.0*U0*(1-pow(4*y - 6*S,2))*(1-pow(2*z-1,2)); //-16.0*(pow((y - 1.5*S),2) + 0.25*(pow(z - 0.5,2))) +1.0;//1*(1-pow((4*y - 6*S),2)*(1-pow((2*z - 1),2)));//-4.0*(pow(y-0.5,2) + pow(z-0.5,2)) + 1.0;
    if(r.u <0){
        r.u = 0.0; //sanity check
    }
    }
    



    return r;

}


inline double  BACKWARDS_FACING_STEP_PRESSURE(double x, double y,double z,double t){

    return 0.0;

}



/*//////////////////////////////////////////////
///////////Identity Functions ///////////////
*///////////////////////////////////////////////

inline Vec3 ZERO(double x, double y,double z,double ){
    Vec3 vec;
    vec.u = 0;
    vec.v = 0;
    vec.w = 0;
    return vec;
}

inline Vec2 ZERO2D(double x, double y,double ){
    Vec2 vec;
    vec.u = 0;
    vec.v = 0;

    return vec;
}


inline double ZERO_SCALAR(double x,double y,double z,double t){
    return 0.0;
}

inline double ZERO2D_SCALAR(double x,double y,double t){
    return 0.0;
}

inline Vec3 ONE(double x, double y,double z,double ){
    Vec3 vec;
    vec.u = 1;
    vec.v = 1;
    vec.w = 1;
    return vec;
}

inline double ONE_SCALAR(double x,double y,double z,double t){
    return 1.0;
}


/*//////////////////////////////////////////////
/////////// Solid Masks ///////////////
*///////////////////////////////////////////////

inline int LID_CAVITY_SOLID_MASK(int i,int j,int k){
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 ||  j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1){
        return SOLID_CELL;
    }
    else{


        return FLUID_CELL;
    }
}

inline int BACKWARDS_FACING_STEP_SOLID_MASK(int i, int j, int k) {
    // First check for empty cell condition
    if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return EMPTY_CELL;
    }
    // Then check for inflow condition
    if(j == 0 && i > SIMULATION.Ny/2 && i != SIMULATION.Ny - 1 &&  k != 0 && k != SIMULATION.Nz-1) {
        return INFLOW_CELL;
    }
    // Finally check for solid boundaries
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1) {
        return SOLID_CELL;
    }
    // Default to fluid cell
    return FLUID_CELL;
}



inline int EMPTY_SOLID_MASK(int  i,int j,int k){
     // First check for empty cell condition
     if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return EMPTY_CELL;
    }

    return FLUID_CELL;



}

/*//////////////////////////////////////////////
///////////Backwards Facing Step 2D ///////////////
*///////////////////////////////////////////////

inline Vec2 BACKWARDS_FACING_STEP_2D(double x, double y,double t){
    Vec2 r;
    r.u = 0.0;
    r.v = 0.0;
    double dh = SIMULATION2D.dh;
    double S = 0.49;
    //on the top of the step

    if(x <= SIMULATION2D.dh  && y <= (1.0 - SIMULATION2D.dh/2.0) && y >= SIMULATION2D.dh/2 && y > S){
        r.u = -16.0*(pow((y - 1.5*S),2)) +1.0;//1*(1-pow((4*y - 6*S),2)*(1-pow((2*z - 1),2)));//-4.0*(pow(y-0.5,2) + pow(z-0.5,2)) + 1.0;
        //r.u = 1.0;
    if(r.u <0){
        r.u = 0.0; //sanity check
    }
    }
    



    return r;

}


inline double  BACKWARDS_FACING_STEP_PRESSURE_2D(double x, double y,double t){

    return 0.0;

}

inline int BACKWARDS_FACING_STEP_SOLID_MASK_2D(int i, int j) {
    // First check for empty cell condition
    if(j == SIMULATION2D.Nx-1 && i != 0 && i != SIMULATION2D.Ny-1 ) {
        return EMPTY_CELL;
    }
    // Then check for inflow condition
    if(j == 0 && i > SIMULATION2D.Ny/2 && i != SIMULATION2D.Ny - 1 ) {
        return INFLOW_CELL;
    }
    // Finally check for solid boundaries
    if(i == 0 || j == 0 || i == SIMULATION2D.Ny-1 || j == SIMULATION2D.Nx-1) {
        return SOLID_CELL;
    }
    // Default to fluid cell
    return FLUID_CELL;
}

/*//////////////////////////////////////////////
///////////OBSTACLE  3D ///////////////
*///////////////////////////////////////////////
inline Vec3 OBSTACLE_FLOW(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    double U0 = 2.25;

    if(x <= SIMULATION.dh  && y < (6.0 - SIMULATION.dh/2.0) && y > SIMULATION.dh/2 && z < (6.0 - SIMULATION.dh/2.0) && z > SIMULATION.dh/2){
        r.u = 16.0*U0*y*z*(0.41 - y) *(0.41-z)/pow(0.41,4);

    }
    if(r.u <0){
        r.u = 0.0;
    }




    return r;

}

inline double  OBSTACLE_FLOW_PRESSURE(double x, double y,double z,double t){

    return 0.0;

}

inline int OBSTACLE_SOLID_MASK(int i, int j, int k) {
    double dh = SIMULATION.dh;
    double x = j * dh;
    double y = i * dh;
    
    // Check for inflow condition FIRST (most specific)
    if(j == 0 && i >= 1 && i < SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return INFLOW_CELL;
    }
    
    // Check for outflow/empty condition
    if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return EMPTY_CELL;
    }
    
    // Check for solid boundaries
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1) {
        return SOLID_CELL;
    }
    
    if(x >= 0.45 && x <= 0.55 && y >= 0.15 && y <= 0.25) {
        return SOLID_CELL;
    }
    
    return FLUID_CELL;
}


/*//////////////////////////////////////////////
///////////OBSTACLE  2D ///////////////
*///////////////////////////////////////////////

inline Vec2 OBSTACLE_FLOW_2D(double x, double y,double t){
    Vec2 r;
    r.u = 0.0;
    r.v = 0.0;


    double U0 = 1.5;

    if(x <= SIMULATION2D.dh  && y < (6.0 - SIMULATION2D.dh/2.0) && y > SIMULATION2D.dh/2){
        r.u = U0 - (U0/(0.205*0.205))*pow((y - 0.205),2);
        //r.u = 1.0;
    }
    if(r.u <0){
        r.u = 0.0;
    }


    return r;

}

inline double  OBSTACLE_FLOW_PRESSURE_2D(double x, double y,double t){

    return 0.0;

}

inline int OBSTACLE_SOLID_MASK_2D(int  i,int j){
     // First check for empty cell condition
     if(j == SIMULATION2D.Nx-1 && i != 0 && i != SIMULATION2D.Ny-1) {
        return EMPTY_CELL;
    }
    // Then check for inflow condition
    if(j == 0 && i >= 1 && i < SIMULATION2D.Ny-1 ) {
        return INFLOW_CELL;
    }
    // Finally check for solid boundaries
    if(i == 0 || j == 0 || i == SIMULATION2D.Ny-1 || j == SIMULATION2D.Nx-1 ) {
        return SOLID_CELL;
    }
    double dh = SIMULATION2D.dh;
    double radius = 0.05;

    if(pow((j*dh - 0.2),2) +pow((i*dh - 0.2),2)  < radius*radius){
        return SOLID_CELL;
    }


    return FLUID_CELL;



}

#endif