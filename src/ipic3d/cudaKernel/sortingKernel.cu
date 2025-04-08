
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"


__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* fillerHashedSum, int numberOfHole){

	const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint gridSize = blockDim.x * gridDim.x;

	for (int pidx = tidx + (pclsArray->getNOP() - numberOfHole); pidx < pclsArray->getNOP(); pidx += gridSize) {
		auto departureElement = departureArray->getArray() + pidx;
		if(departureElement->dest != 0)continue; 		// exiting particles, the holes in the rear part

		auto pcl = pclsArray->getpcls() + pidx;

		auto index = fillerHashedSum->getIndex(pidx, departureElement->hashedId); // updated

		fillerBuffer->getArray()[index] = pidx;
	}

}




__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* holeHashedSum, int numberOfHole){

	const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint gridSize = blockDim.x * gridDim.x;

	for (int pidx = tidx; pidx < (pclsArray->getNOP() - numberOfHole); pidx += gridSize) {

		auto departureElement = departureArray->getArray() + pidx;
		if(departureElement->dest == 0)continue; 				// exiting particles, the holes

		auto pcl = pclsArray->getpcls() + pidx;

		auto index = holeHashedSum->getIndex(pidx, departureElement->hashedId); // updated

		memcpy(pcl, pclsArray->getpcls() + fillerBuffer->getArray()[index], sizeof(SpeciesParticle));

	}


}







