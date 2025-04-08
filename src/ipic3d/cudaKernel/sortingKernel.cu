
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"


__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* fillerHashedSum, int numberOfHole){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint gridSize = blockDim.x * gridDim.x;

	pidx += (pclsArray->getNOP() - numberOfHole);			// rear part of pclArray

	for (int i = pidx; i < pclsArray->getNOP(); i += gridSize) {
		auto departureElement = departureArray->getArray() + i;
		if(departureElement->dest != 0)continue; 		// exiting particles, the holes in the rear part

		auto pcl = pclsArray->getpcls() + i;

		auto index = fillerHashedSum->getIndex(i, departureElement->hashedId); // updated

		fillerBuffer->getArray()[index] = i;
	}

}




__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* holeHashedSum, int numberOfHole){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint gridSize = blockDim.x * gridDim.x;

	for (int i = pidx; i < pclsArray->getNOP() - numberOfHole; i += gridSize) {

		auto departureElement = departureArray->getArray() + i;
		if(departureElement->dest == 0)continue; 				// exiting particles, the holes

		auto pcl = pclsArray->getpcls() + i;

		auto index = holeHashedSum->getIndex(i, departureElement->hashedId); // updated

		memcpy(pcl, pclsArray->getpcls() + fillerBuffer->getArray()[index], sizeof(SpeciesParticle));

	}


}







