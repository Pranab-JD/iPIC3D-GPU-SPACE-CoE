
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"



/**
 * @brief   Copy the exiting particles of the species to ExitingBuffer. Launch (nop) threads.
 * @details By the end of Mover, the DepartureArray has been prepared, host allocates the ExitingBuffer according 
 *          to the Hashed SumUp value for each direction.
 *          This kernel is responsible for moving the exiting particles in the pclArray into the ExitingBuffer,
 *          according to the DepartureArray.  
 *          The exiting particles in ExitingBuffer are orginaized in their destinations, with random order adopted from hashedSum.
 *          
 *          This kernel is also responsible for preparing the 2 hashedSum for SortingKernel1 and SortingKernel2. It will modify the 
 *          elements of the departure array.
 *          
 * @param exitingArray The buffer used for exiting particles for 6 directions, the size and distriburtion are decided by the 6 hashedSum
 * @param hashedSumArray 9 hashedSum, 6 from the Mover, 1 for the deleted, 2 for Sorting.
 * 
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                exitingArray* exitingArray, hashedSum* hashedSumArray){
                                    
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= pclsArray->getNOP())return;

    __shared__ int x; // y, the number of holes (eixitng + deleted)
    if(threadIdx.x == 0){ 
        x = 0; 
        for(int i=0; i <= departureArrayElementType::DELETE_HASHEDSUM_INDEX ; i++)x += hashedSumArray[i].getSum(); 
    }
    __syncthreads();
    
    auto departureElement = departureArray->getArray() + pidx;
    // return the stayed particles in the front part
    if(pidx < (pclsArray->getNOP()-x) && departureElement->dest == 0)return; 
    
    // Exiting particles
    if(departureElement->dest > 0 && departureElement->dest < departureArrayElementType::DELETE){ 
        auto pcl = pclsArray->getpcls() + pidx;

        int index = 0;
        // get the index in exitingBuffer
        for(int i=0; i < departureElement->dest-1; i++){
            index += hashedSumArray[i].getSum(); // compact exiting buffer
        }
        // index in its direction
        index += hashedSumArray[departureElement->dest-1].getIndex(pidx, departureElement->hashedId);
        // copy the particle
        memcpy(exitingArray->getArray() + index, pcl, sizeof(SpeciesParticle));
    }

    // holes
    if(departureElement->dest !=0){
        if(pidx >= (pclsArray->getNOP()-x))return; // return holes in the back part
        departureElement->hashedId = hashedSumArray[departureArrayElementType::HOLE_HASHEDSUM_INDEX].add(pidx);
        return; // return all holes
    }

    // Only fillers reach here
    if(pidx >= (pclsArray->getNOP()-x)){ 
        departureElement->hashedId = hashedSumArray[departureArrayElementType::FILLER_HASHEDSUM_INDEX].add(pidx);
    }

}







