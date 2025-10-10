#ifndef _MOMENTKERNEL_CUH_
#define _MOMENTKERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"

class momentParameter
{
public:

    particleArrayCUDA* pclsArray; // default main array

    departureArrayType* departureArray; // a helper array for marking exiting particles

    cudaParticleType dt;
    cudaParticleType qom;
    cudaParticleType c;

    //! @param pclsArrayCUDAPtr - it should be a device pointer
    __host__ momentParameter(particleArrayCUDA* pclsArrayCUDAPtr, departureArrayType* departureArrayCUDAPtr)
    {
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
    }
};

/**
 *! Moment kernel:
 *!     - one particle per thread is fine
 *!     - Should be launched in species --- if there are 4 species, launch 4 times in different streams
 * 
 * @param grid 
 * @param _pcls the particles of a species
 * @param moments array4, [x][y][z][density], 
 *                  here[nxn][nyn][nzn][10], must be 0 before kernel launch
 */

__global__ void momentKernelStayed(momentParameter* momentParam,
                                   grid3DCUDA* grid,
                                   cudaTypeArray1<cudaMomentType> moments);

__global__ void momentKernelNew(momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle);

__global__ void ECSIM_RelSIM_Moments_PreExchange( momentParameter* momentParam,
                                                  grid3DCUDA* grid,
                                                  cudaTypeArray1<cudaFieldType> fieldForPcls,
                                                  cudaTypeArray1<cudaMomentType> moments);

__global__ void ECSIM_RelSIM_Moments_PostExchange(momentParameter* momentParam,
                                                  grid3DCUDA* grid,
                                                  cudaTypeArray1<cudaFieldType> fieldForPcls,
                                                  cudaTypeArray1<cudaMomentType> moments,
                                                  int stayedParticle);

__device__ inline void exact_mass_matrix(cudaTypeArray1<cudaMomentType> moments,
                                        int ix, int iy, int iz,
                                        double q, double q_dt_2mc,
                                        const double weights[8],
                                        double a00, double a01, double a02,
                                        double a10, double a11, double a12,
                                        double a20, double a21, double a22,
                                        int nxn, int nyn, int nzn);

#endif