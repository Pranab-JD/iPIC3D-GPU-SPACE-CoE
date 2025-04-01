#ifndef _PARTICLE_CONTROL_KERNEL_CUH_
#define _PARTICLE_CONTROL_KERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleControlKernel.cuh"

#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

#include "gridCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"
#include "moverKernel.cuh"


__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList, grid3DCUDA* grid, particleArrayCUDA* pclArray, departureArrayType* departureArray);

template <bool MULTIPLE>
__global__ void particleSplittingKernel(moverParameter* moverParam, grid3DCUDA* grid);

#endif // _PARTICLE_CONTROL_KERNEL_CUH_
