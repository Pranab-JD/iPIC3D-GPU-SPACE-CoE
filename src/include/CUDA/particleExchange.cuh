#ifndef _PARTICLE_EXCHANGE_H_
#define _PARTICLE_EXCHANGE_H_

#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"


typedef struct departureArrayElement_s{
    enum departureDestination{
        STAY = 0,

        XLOW = 1,
        XHIGH,
        YLOW,
        YHIGH,
        ZLOW,
        ZHIGH,

        DELETE = 7
    };

    uint32_t dest;          // destination of the particle exchange
    uint32_t hashedId;      // the id got from hashed sum
}departureArrayElement_t;

using departureArrayElementType = departureArrayElement_t;
using departureArrayType = arrayCUDA<departureArrayElementType>;

inline constexpr int HASHED_SUM_NUM = departureArrayElementType::DELETE+2;

using exitingArray = arrayCUDA<SpeciesParticle>;

using fillerBuffer = arrayCUDA<int>;


__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                exitingArray* exitingArray, hashedSum* hashedSumArray);

__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int x);

__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int stayedParticle);


#endif