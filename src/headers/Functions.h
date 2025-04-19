#ifndef EXACT_H
#define EXACT_H
#include "Utils.h"
#include "math.h"
#include "Definitions.h"




/*//////////////////////////////////////////////
////////3D Taylor Green - Not Working///////////
*///////////////////////////////////////////////

inline double f_x_TGV(double x, double y, double z, double t) {

    const double sin_x = std::sin(x);
    const double sin_y = std::sin(y);
    const double sin_z = std::sin(z);
    const double cos_x = std::cos(x);
    
    const double angle = 2.0 * sin_x * sin_x + 2.0 * sin_y * sin_y - 2.0;
    
    const double term1 = 0.5 * sin_z * sin_z * std::cos(angle);
    const double term2 = 1.0 * sin_z * sin_z;
    const double term3 = 0.75 * std::cos(angle);
    
    const double part1 = SIMULATION.RE * (term1 - term2 - term3) * sin_x;
    const double part2 = -6.0 * sin_y * sin_y * sin_z * sin_z * cos_x;
    
    return (part1 + part2) * std::exp(-4.0 * t / SIMULATION.RE) * cos_x / SIMULATION.RE;
}

inline double f_y_TGV(double x, double y, double z, double t) {
    const double sin_x = std::sin(x);
    const double sin_y = std::sin(y);
    const double sin_z = std::sin(z);
    const double cos_y = std::cos(y);
    
    const double angle = 2.0 * sin_x * sin_x + 2.0 * sin_y * sin_y - 2.0;
    
    const double term1 = 0.5 * sin_z * sin_z * std::cos(angle);
    const double term2 = 1.0 * sin_z * sin_z;
    const double term3 = 0.75 * std::cos(angle);
    
    const double part1 = SIMULATION.RE * (term1 - term2 - term3) * sin_y;
    const double part2 = -6.0 * sin_x * sin_x * sin_z * sin_z * cos_y;
    
    return (part1 + part2) * std::exp(-4.0 * t / SIMULATION.RE) * cos_y / SIMULATION.RE;
}

inline double f_z_TGV(double x, double y, double z, double t) {
    const double sin_2z = std::sin(2.0 * z);
    const double cos_2x = std::cos(2.0 * x);
    const double cos_2y = std::cos(2.0 * y);
    
    return -0.125 * std::exp(-4.0 * t / SIMULATION.RE) * sin_2z * std::sin(cos_2x + cos_2y);
}


inline Vec3 TAYLOR_GREEN_VORTEX_VELOCITY(double x,double y,double z,double t){

    Vec3 r;
    r.u =  exp(-2*t/SIMULATION.RE)*cos(x)*sin(y)*sin(z);
    r.v = -exp(-2*t/SIMULATION.RE)*1*sin(x)*cos(y)*sin(z);
    r.w = 0.0;  



    return r;

}

inline double TAYLOR_GREEN_VORTEX_PRESSURE(double x,double y,double z,double t){

    return exp(-(4*t*t/SIMULATION.RE))*((1.0/16.0) * (sin(cos(2*x) + cos(2*y))*cos(2*z) + 2.0));

}


/*//////////////////////////////////////////////
//////////////2D Taylor Green //////////////////
*///////////////////////////////////////////////
inline Vec3 TAYLOR_GREEN_VORTEX_VELOCITY_2D(double x,double y,double z,double t){
    Vec3 r;
    r.w = 0.0;

    r.u = exp(-2*t /SIMULATION.RE)*sin(x)*cos(y);
    r.v = -exp(-2*t /SIMULATION.RE)*cos(x)*sin(y);
    return r;
    

}

inline double TAYLOR_GREEN_VORTEX_PRESSURE_2D(double x,double y,double z,double t){
    return 0.25* (cos(2*x) + cos(2*y))*exp(-4*t /SIMULATION.RE);
}

inline double f_x_TGV_2D(double x, double y, double z, double t){
    return -0.5 * exp(-4*t/SIMULATION.RE) * sin(2*x);
}
inline double f_y_TGV_2D(double x, double y, double z, double t){
    return -0.5 * exp(-4*t/SIMULATION.RE) * sin(2*y);
}
inline double f_z_TGV_2D(double x, double y, double z, double t){
    return 0.0;
}

inline Vec3 TAYLOR_GREEN_FONT(double x,double y,double z,double t){
    Vec3 vec;
    vec.u = f_x_TGV_2D(x,y,z,t);
    vec.v = f_y_TGV_2D(x,y,z,t);
    vec.w = f_z_TGV_2D(x,y,z,t);
    return vec;
}

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
    //on the top of the step

    if(x <= SIMULATION.dh  && (z<=1-SIMULATION.dh/2.0) && z>=SIMULATION.dh/2.0 && y <= (1.0 - SIMULATION.dh/2.0) && y >= SIMULATION.dh/2 && y > S){
        r.u = -16.0*(pow((y - 1.5*S),2) + 0.25*(pow(z - 0.5,2))) +1.0;//1*(1-pow((4*y - 6*S),2)*(1-pow((2*z - 1),2)));//-4.0*(pow(y-0.5,2) + pow(z-0.5,2)) + 1.0;
        //r.u = 1.0;
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
///////////////Obstacle flow////////////////////
*///////////////////////////////////////////////

inline Vec3 OBSTACLE_FLOW(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    if(x <= SIMULATION.dh  && (z<1-SIMULATION.dh/2.0) && z>SIMULATION.dh/2.0 && y < (1.0 - SIMULATION.dh/2.0) && y > SIMULATION.dh/2){
        r.u = (1.0 - pow(2.0*y - 1.0, 2)) * (1.0 - pow(2.0*z - 1.0, 2));//-4.0*(pow(y-0.5,2) + pow(z-0.5,2)) + 1.0;
        //r.u = 1.0;
    }
    if(r.u <0){
        r.u = 0.0;
    }


    return r;

}

inline double  OBSTACLE_FLOW_PRESSURE(double x, double y,double z,double t){

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

inline double ZERO_SCALAR(double x,double y,double z,double t){
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

inline int OBSTACLE_SOLID_MASK(int  i,int j,int k){
     // First check for empty cell condition
     if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return EMPTY_CELL;
    }
    // Then check for inflow condition
    if(j == 0 && i >= 1 && i < SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return INFLOW_CELL;
    }
    // Finally check for solid boundaries
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1) {
        return SOLID_CELL;
    }
    double dh = SIMULATION.dh;
    double radius = 0.25;
    //finally, check foor the obstacle, which is a r=0.25 sphere centered at 0.5,0.5,0.5
    //sphere
    if(pow((j*dh - 0.5),2) +pow((i*dh - 0.5),2)+ pow((k*dh - 0.5),2)  < radius*radius){
        return SOLID_CELL;
    }
    //cilinder
    //if(pow((j*dh - 1.0),2) +pow((i*dh - 0.5),2) < radius*radius && (k*dh)>=0.25 && (k*dh)<=0.75){
    //    return SOLID_CELL;
    //}
    //elipse
    // Rotated ellipse with 3:1 ratio (x:z) rotated by 10 degrees
    //float center_x = 2.0;
    //float center_z = 0.5;
    //float angle = 0 * M_PI / 180.0; // Convert degrees to radians
    //float cos_angle = cos(angle);
    //float sin_angle = sin(angle);
//
    //// Translate coordinates to origin, rotate, then translate back
    //float x = j*dh - center_x;
    //float z = i*dh - center_z;
    //float x_rot = x * cos_angle - z * sin_angle;
    //float z_rot = x * sin_angle + z * cos_angle;
//
    //// Check ellipse equation (3:1 ratio) and y bounds
    //if(pow(x_rot/radius, 2) + pow(z_rot/(radius), 2) < 1.0 && 
    //   (k*dh)>=0.25 && (k*dh)<=0.75){
    //    return SOLID_CELL;
    //}
    //if(pow((j*dh - 0.5),2) +pow((i*dh - 0.5),2)+ pow((k*dh - 0.5),2)  < (dh*1.0)*(dh*1.0)){
    //    return SOLID_CELL;
    //}
    // Default to fluid cell
    return FLUID_CELL;



}



#endif