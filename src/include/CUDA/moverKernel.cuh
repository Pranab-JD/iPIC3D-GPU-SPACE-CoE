#ifndef _MOVERKERNEL_CUH_
#define _MOVERKERNEL_CUH_

#include "Particles3D.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "EMfields3D.h"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"

using namespace std;

//? RelSIM particle pusher
enum Relativistic_pusher : uint8_t { BORIS = 0, LAPENTA_MARKIDIS = 1 };

class moverParameter
{

public: //particle arrays

    particleArrayCUDA* pclsArray;           // default main array
    departureArrayType* departureArray;     // a helper array for marking exiting particles
    hashedSum* hashedSumArray;              // 8 hashed sum

public: // common parameter

    cudaParticleType dt;
    cudaParticleType qom;
    cudaParticleType c;

    int NiterMover;
    int DFIELD_3or4;
    cudaParticleType umax, umin, vmax, vmin, wmax, wmin;

    //? Relativistic toggle (0 = Non Relativistic, 1 = Relativistic)
    uint8_t Relativistic;

    //? RelSIM particle pusher (used when Relativistic == 1)
    uint8_t relativistic_pusher;          // RelPusher::BORIS or RelPusher::LAPENTA_MARKIDIS

    //? Boundary Conditions
    // For openBC, XLeft, XRight, YLeft, YRight, ZLeft, ZRight
    bool doOpenBC; // a OR of applyOpenBC
    bool applyOpenBC[6];
    cudaCommonType deleteBoundary[6];
    cudaCommonType openBoundary[6];
    uint32_t appendCountAtomic; // the number of duplicated particles to be appended to the array, just in time

    // For repopulate injection, XLeft, XRight, YLeft, YRight, ZLeft, ZRight
    bool doRepopulateInjection;
    bool doRepopulateInjectionSide[6];
    cudaCommonType repopulateBoundary[6];

    // For sphere
    int doSphere; // 0: no sphere, 1: sphere, 2: sphere2D(XZ)
    cudaCommonType sphereOrigin[3];
    cudaCommonType sphereRadius;

public:

    __host__ moverParameter(Particles3D* p3D, VirtualTopology3D* vct)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = particleArrayCUDA(p3D).copyToDevice();
        departureArray = departureArrayType(p3D->getNOP() * 1.5).copyToDevice();
    }

    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ moverParameter(Particles3D* p3D, particleArrayCUDA* pclsArrayCUDAPtr, 
                            departureArrayType* departureArrayCUDAPtr, hashedSum* hashedSumArrayCUDAPtr)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
        hashedSumArray = hashedSumArrayCUDAPtr;
    }
};

//! IMM
__global__ void moverKernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid);

//! IMM with adaptive subcycling --> divides dt by eight times the particle gyroperiod and performs a relativistic velocity update
__global__ void moverSubcyclesKernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid);

__global__ void updatePclNumAfterMoverKernel(moverParameter *moverParam);

//! ECSIM
// __global__ void ECSIM_position_kernel(moverParameter *moverParam, cudaTypeArray1<commonType> resDivC, grid3DCUDA *grid);
__global__ void ECSIM_position_kernel(moverParameter *moverParam, grid3DCUDA *grid);    //TODO: Charge conservation is to be implemented
__global__ void ECSIM_velocity_kernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid);

//! RelSIM
__global__ void RelSIM_velocity_kernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grids);

// __global__ void castingField(grid3DCUDA *grid, cudaTypeArray1<cudaCommonType> fieldForPcls);


#endif