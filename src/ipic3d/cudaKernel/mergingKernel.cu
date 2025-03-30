

#include "cudaTypeDef.cuh"
#include "mergingKernel.cuh"

#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"


/**
* @brief Merging kernel, merging particle pairs with similar velocity in the same cell
*
* @param cellOffsetList absolute Offset of the cells in the pclArray, the pclArray must be sorted
* @param cellBinCountList Number of particles in each cell
* @param grid Pointer to the grid structure
* @param pcl Pointer to the particle array
* @param departureArray Pointer to the departure array, for delete mark
* @details one warp for each cell
*/
__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList, int cellNum, grid3DCUDA* grid, particleArrayCUDA* pclArray, departureArrayType* departureArray) {

    const uint pid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint warpId = pid / WARP_SIZE; const auto& cellId = warpId;
    const uint laneId = pid % WARP_SIZE;

    // return if pid > number of particle rounded up to warpsize
    const int nop = pclArray->getNOP();
    auto pcl = pclArray->getpcls();
    auto dArray = departureArray->getArray();

    if (cellId >= cellNum) return;

    // cell offset for this warp
    const int cellOffset = cellOffsetList[cellId];
    // number of particles in this cell
    const int numPIC = cellBinCountList[cellId];

    if(numPIC < 32) return; // no merging if less than 32 particles in the cell

    // main loop for one cell, pushing right
    for(int p=0; p<numPIC; p++) {
        const int mainPId = cellOffset + p;

        constexpr cudaParticleType threshold = 0.0005; // threshold for merging
        cudaParticleType minNorm = 1e10; // minimum norm
        int minPId = -1; // minimum particle id

        // each thread calculate the VV norm between the particles it holds and the main loop particle
        // keep the smallest one
        for (int i=laneId; i<numPIC; i+=WARP_SIZE) {
            const int pId = cellOffset + i;
            if (pId <= mainPId) continue; // ignore self, and the past particles

            // ignore if the particle is marked for deletion
            if (dArray[pId].dest != 0) continue;

            // calculate the VV norm
            const auto& p1 = pcl[mainPId];
            const auto& p2 = pcl[pId];

            const auto& u1 = p1.get_u();
            const auto& v1 = p1.get_v();
            const auto& w1 = p1.get_w();

            const auto& u2 = p2.get_u();
            const auto& v2 = p2.get_v();
            const auto& w2 = p2.get_w();

            const auto norm = (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2) + (w1 - w2) * (w1 - w2); // not distance, reduce the sqrt

            if (norm < minNorm) {
                minNorm = norm;
                minPId = pId;
            }
        }

        // warp reduce
        auto localNorm = minNorm;
        auto localPId = minPId;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            auto otherVal = __shfl_down_sync(WARP_FULL_MASK, localNorm, offset);
            auto otherLane = __shfl_down_sync(WARP_FULL_MASK, localPId, offset);

            if (otherVal < localNorm) {
                localNorm = otherVal;
                localPId = otherLane;
            }

        }

        // lane 0 holds the minimum norm
        if (laneId == 0) {
            if (localNorm < threshold) { // merge!
                dArray[localPId].dest = departureArrayElementType::DELETE;

                SpeciesParticle mergedParticle;

                const auto& p1 = pcl[mainPId];
                const auto& p2 = pcl[localPId];
    
                const auto& u1 = p1.get_u();
                const auto& v1 = p1.get_v();
                const auto& w1 = p1.get_w();
                const auto& q1 = p1.get_q();
                const auto& x1 = p1.get_x();
                const auto& y1 = p1.get_y();
                const auto& z1 = p1.get_z();
    
                const auto& u2 = p2.get_u();
                const auto& v2 = p2.get_v();
                const auto& w2 = p2.get_w();
                const auto& q2 = p2.get_q();
                const auto& x2 = p2.get_x();
                const auto& y2 = p2.get_y();
                const auto& z2 = p2.get_z();

                const auto newQ = q1 + q2;
                mergedParticle.set_u((u1*q1 + u2*q2) / newQ);
                mergedParticle.set_v((v1*q1 + v2*q2) / newQ);
                mergedParticle.set_w((w1*q1 + w2*q2) / newQ);
                mergedParticle.set_q(newQ);
                mergedParticle.set_x((x1*q1 + x2*q2) / newQ);
                mergedParticle.set_y((y1*q1 + y2*q2) / newQ);
                mergedParticle.set_z((z1*q1 + z2*q2) / newQ);

                
                pcl[mainPId] = mergedParticle; // merge the particles

            }
        }
            
    }



}