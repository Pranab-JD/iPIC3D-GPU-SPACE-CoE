

#include "cudaTypeDef.cuh"
#include "particleControlKernel.cuh"

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
__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList, grid3DCUDA* grid, particleArrayCUDA* pclArray, departureArrayType* departureArray) {

    const uint pid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint warpId = pid / WARP_SIZE; const auto& cellId = warpId;
    const uint laneId = pid % WARP_SIZE;

    // return if pid > number of particle rounded up to warpsize
    const int nop = pclArray->getNOP();
    auto pcl = pclArray->getpcls();
    auto dArray = departureArray->getArray();
    const uint cellNum = ((grid->nxc) * (grid->nyc) * (grid->nzc));
    if (cellId >= cellNum) return;

    // cell offset for this warp
    const int cellOffset = cellOffsetList[cellId];
    // number of particles in this cell
    const int numPIC = cellBinCountList[cellId];

    const int initialPIC = pclArray->getInitialNOP() / ((grid->nxc-2) * (grid->nyc-2) * (grid->nzc-2));

    if(numPIC <= initialPIC) return; // no merging if less than 32 particles in the cell

    int cellMergeCount = 0; // number of particles merged in this cell

    // main loop for one cell, pushing right
    for(int p=0; p<numPIC; p++) {
        const int mainPId = cellOffset + p;

        constexpr cudaParticleType threshold = 0.009; // threshold for merging, percentage of the velocity of the main particle
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
            const auto& p1 = pcl[mainPId];
            const auto& u1 = p1.get_u();
            const auto& v1 = p1.get_v();
            const auto& w1 = p1.get_w();

            if (localNorm < (threshold * (u1*u1 + v1*v1 + w1*w1))) { // merge!
            //if (true) { // merge!
                dArray[localPId].dest = departureArrayElementType::DELETE;

                SpeciesParticle mergedParticle;

                const auto& p2 = pcl[localPId];
    
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

                cellMergeCount++; 

            }
        }
        // return this warp if reaches the initial number of particles 
        cellMergeCount = __shfl_sync(WARP_FULL_MASK, cellMergeCount, 0); 
        if ((numPIC-cellMergeCount) <= initialPIC) return; 
    }

}

using commonType = cudaParticleType;

// Particle number control 

// Particle splitting kernel to use when the number of particles to be generated is < number available particles
// launch the kernel with number of threads = deltaPcl --> each thread splits a particle randomly choosen
template <>
__global__ void particleSplittingKernel<false>(moverParameter* moverParam, grid3DCUDA* grid)
{   
    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = moverParam->pclsArray;
    const uint deltaPcl = pclsArray->getInitialNOP() - pclsArray->getNOP();
    if(tidx >= deltaPcl)return;

    // grid properties
    const commonType& inv_dx = grid->invdx;
    const commonType& inv_dy = grid->invdy;
    const commonType& inv_dz = grid->invdz;
    const commonType& xstart = grid->xStart;
    const commonType& ystart = grid->yStart;
    const commonType& zstart = grid->zStart;
    
    // batch must be >= 1 --> there is no safety check, it must be checked before launching the kernel
    const uint batch = pclsArray->getNOP() / deltaPcl;
    
    // generate random idx in [0,batch-1] to select the particle to split 
    // based on LCRNG
    uint idxRNG = 0;
    if(batch > 1){
        const uint seed = (1313492u + tidx);
        //seed ^= (seed >> 7);  
        idxRNG = ( seed * deltaPcl + batch - 1 ) % batch;
    }
    // select particle to split
    const uint pidx = tidx * batch + idxRNG;

    // copy the particle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;
    SpeciesParticle newPcl = *pcl;
    const auto x0 = pcl->get_x();
    const auto y0 = pcl->get_y();
    const auto z0 = pcl->get_z();

    // index of the grid point to the right of the particle
    const int ix = 2 + int(floor((x0 - xstart) * inv_dx));
    const int iy = 2 + int(floor((y0 - ystart) * inv_dy));
    const int iz = 2 + int(floor((z0 - zstart) * inv_dz));
    // distance particle - grid point to the left
    const commonType xi0 = x0 - grid->getXN(ix - 1);
    const commonType yi0 = y0 - grid->getYN(iy - 1);
    const commonType zi0 = z0 - grid->getZN(iz - 1);
    // distance particle - grid point to the right
    const commonType xi1 = grid->getXN(ix) - x0;
    const commonType yi1 = grid->getYN(iy) - y0;
    const commonType zi1 = grid->getZN(iz) - z0;

    // select the lowest distance to ensure keeping the particles in the cell
    cudaTypeDouble delta = xi0;
    if (yi0 < delta) delta = yi0;
    if (zi0 < delta) delta = zi0;
    if (xi1 < delta) delta = xi1;
    if (yi1 < delta) delta = yi1;
    if (zi1 < delta) delta = zi1;
            
    delta /= 20;
    pcl->set_x(x0 - delta);
    pcl->set_y(y0 - delta);
    pcl->set_z(z0 - delta);
    newPcl.set_x(x0 + delta);
    newPcl.set_y(y0 + delta);
    newPcl.set_z(z0 + delta);
    
    // update weights
    const auto q = pcl->get_q(); 
    pcl->set_q( 0.5 * q );
    newPcl.set_q( 0.5 * q );
    
    newPcl.set_t(114515.0);

    const auto index = pclsArray->getNOP() + tidx;
    // check memory overflow
    if (index >= moverParam->pclsArray->getSize()) {
        printf("Memory overflow in open boundary outflow\n");
        //__trap();
        return;
    }
    memcpy(moverParam->pclsArray->getpcls() + index, &newPcl, sizeof(SpeciesParticle));
}




// Particle splitting kernel to use when the number of particles to be generated is > number available particles
// launch the kernel with number of threads = pclsArray->getNOP() --> each thread splits a particle splittingTimes times

template <>
__global__ void particleSplittingKernel<true>(moverParameter* moverParam, grid3DCUDA* grid)
{  
    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = moverParam->pclsArray;
    if(tidx >= pclsArray->getNOP())return;

    const uint deltaPcl = pclsArray->getInitialNOP() - pclsArray->getNOP();

    // it is assumed deltaPcl >= pclsArray->getNOP() - no safety check
    const uint splittingTimes = deltaPcl / pclsArray->getNOP();

    // grid properties
    const commonType& inv_dx = grid->invdx;
    const commonType& inv_dy = grid->invdy;
    const commonType& inv_dz = grid->invdz;
    const commonType& xstart = grid->xStart;
    const commonType& ystart = grid->yStart;
    const commonType& zstart = grid->zStart;
    
    for(int i = 0; i < splittingTimes; i ++)
    {
        const uint pidx = i * pclsArray->getNOP() + tidx;
        // copy the particle
        SpeciesParticle *pcl = pclsArray->getpcls() + pidx;
        SpeciesParticle newPcl = *pcl;

        const auto x0 = pcl->get_x();
        const auto y0 = pcl->get_y();
        const auto z0 = pcl->get_z();

        // index of the grid point to the right of the particle
        const int ix = 2 + int(floor((x0 - xstart) * inv_dx));
        const int iy = 2 + int(floor((y0 - ystart) * inv_dy));
        const int iz = 2 + int(floor((z0 - zstart) * inv_dz));
        // distance particle - grid point to the left
        const commonType xi0 = x0 - grid->getXN(ix - 1);
        const commonType yi0 = y0 - grid->getYN(iy - 1);
        const commonType zi0 = z0 - grid->getZN(iz - 1);
        // distance particle - grid point to the right
        const commonType xi1 = grid->getXN(ix) - x0;
        const commonType yi1 = grid->getYN(iy) - y0;
        const commonType zi1 = grid->getZN(iz) - z0;

        // select the lowest distance to ensure keeping the particles in the cell
        cudaTypeDouble delta = xi0;
        if (yi0 < delta) delta = yi0;
        if (zi0 < delta) delta = zi0;
        if (xi1 < delta) delta = xi1;
        if (yi1 < delta) delta = yi1;
        if (zi1 < delta) delta = zi1;
                
        delta /= 20;
        pcl->set_x(x0 - delta);
        pcl->set_y(y0 - delta);
        pcl->set_z(z0 - delta);
        newPcl.set_x(x0 + delta);
        newPcl.set_y(y0 + delta);
        newPcl.set_z(z0 + delta);
        
        
        const auto q = pcl->get_q(); 
        pcl->set_q( 0.5 * q );
        newPcl.set_q( 0.5 * q );
        
        newPcl.set_t(114515.0);

        const auto index = pclsArray->getNOP() + pidx;
        // check memory overflow
        if (index >= moverParam->pclsArray->getSize()) {
            printf("Memory overflow in open boundary outflow\n");
            //__trap();
            return;
        }
        memcpy(moverParam->pclsArray->getpcls() + index, &newPcl, sizeof(SpeciesParticle));
    }

}