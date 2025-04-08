#ifndef _MOMENTKERNEL_CUH_
#define _MOMENTKERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"





class momentParameter{

public:

    particleArrayCUDA* pclsArray; // default main array

    departureArrayType* departureArray; // a helper array for marking exiting particles


    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ momentParameter(particleArrayCUDA* pclsArrayCUDAPtr, departureArrayType* departureArrayCUDAPtr){
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
    }

};


/**
 * @brief moment kernel, one particle per thread is fine
 * @details the moment kernel should be launched in species
 *          if these're 4 species, launch 4 times in different streams
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


#endif