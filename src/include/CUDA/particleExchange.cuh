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

    enum hashedSumIndex{
        XLOW_HASHEDSUM_INDEX = 0,
        XHIGH_HASHEDSUM_INDEX,
        YLOW_HASHEDSUM_INDEX,
        YHIGH_HASHEDSUM_INDEX,
        ZLOW_HASHEDSUM_INDEX,
        ZHIGH_HASHEDSUM_INDEX,
        DELETE_HASHEDSUM_INDEX = 6,
        HOLE_HASHEDSUM_INDEX,
        FILLER_HASHEDSUM_INDEX,

        HASHED_SUM_NUM = 9
    };

    uint32_t dest;          // destination of the particle exchange
    uint32_t hashedId;      // the id got from hashed sum
}departureArrayElement_t;

using departureArrayElementType = departureArrayElement_t;
using departureArrayType = arrayCUDA<departureArrayElementType>;

using exitingArray = arrayCUDA<SpeciesParticle>;

using fillerBuffer = arrayCUDA<int>;


/**
 * @brief   Copy the exiting particles of the species to ExitingBuffer. 
 * @details By the end of Mover, the DepartureArray has been prepared, host allocates the ExitingBuffer according 
 *          to the Hashed SumUp value for each direction.
 *          This kernel is responsible for moving the exiting particles in the pclArray into the ExitingBuffer,
 *          according to the DepartureArray.  
 *          The exiting particles in ExitingBuffer are orginaized in their destinations, with random order adopted from hashedSum.
 *          
 *          This kernel is also responsible for preparing the 2 hashedSum for SortingKernel1 and SortingKernel2. It will modify the 
 *          elements of the departure array.
 * 
 *          Launch less than (nop) threads, coarsening.
 *          
 * @param exitingArray The buffer used for exiting particles for 6 directions, the size and distriburtion are decided by the 6 hashedSum
 * @param hashedSumArray 9 hashedSum, 6 from the Mover, 1 for the deleted, 2 for Sorting.
 * 
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
    exitingArray* exitingArray, hashedSum* hashedSumArray, int numberOfHole);


/**
 * @brief 	Prepare the filler buffer, making it compact. 
 * @details The exiting particles have been copied to exitingBuffer, together eith the deleted, they left many holes.
 *          It's better to launch numberOfHole threads.
 * 			
 * @param fillerBuffer  Used for storing the index of the stayed particles in between (nop-numberOfHole) to (nop-1), fillers
 * @param fillerHashedSum it contains 1 hashedSum instance, prepared in exitingKernel. 
 * 							For the filler particles in rear part.
 */
__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
    fillerBuffer* fillerBuffer, hashedSum* fillerHashedSum, int numberOfHole);



/**
 * @brief 	Fill the holes in pclArray, making it compact. Launch smaller number of threads, coarsening.
 * @details Sorting2 has to be Launched after sorting1 kernel.
 * 			The indexes of the filler particles have been recorded into the filler buffer in the previous Sorting1Kernel
 * 			
 * @param holeHashedSum it contains 1 hashedSum instance, prepared in exitingKernel. 
 * 							For the exiting, deleted particles in front part.
 */
__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
    fillerBuffer* fillerBuffer, hashedSum* holeHashedSum, int numberOfHole);


#endif