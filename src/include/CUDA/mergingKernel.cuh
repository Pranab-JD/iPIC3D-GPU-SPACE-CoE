#ifndef _MERGING_KERNEL_CUH_
#define _MERGING_KERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "mergingKernel.cuh"

#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList, int cellNum, grid3DCUDA* grid, particleArrayCUDA* pclArray, departureArrayType* departureArray);



#endif // _MERGING_KERNEL_CUH_
