
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"




__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                exitingArray* exitingArray, hashedSum* hashedSumArray, int numberOfHole){
                                    
    uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gridSize = blockDim.x * gridDim.x;
    const uint stayedParticles = pclsArray->getNOP() - numberOfHole;


    for (int pidx = tidx; pidx < pclsArray->getNOP(); pidx += gridSize) {
    
        auto departureElement = departureArray->getArray() + pidx;
        if(pidx < stayedParticles && departureElement->dest == 0)continue;
        
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
            if(pidx >= stayedParticles)continue; // holes in the rear part
            departureElement->hashedId = hashedSumArray[departureArrayElementType::HOLE_HASHEDSUM_INDEX].add(pidx);
            continue; // all holes
        }

        // Only fillers reach here
        if(pidx >= stayedParticles){ 
            departureElement->hashedId = hashedSumArray[departureArrayElementType::FILLER_HASHEDSUM_INDEX].add(pidx);
        }

    }

}







