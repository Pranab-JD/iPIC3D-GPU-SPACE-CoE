
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

namespace velocityHistogram
{

using namespace particleArraySoA;

/**
 * @brief Kernel function to compute velocity histograms.
 * @details launched for each particle
 *
 * @param nop Number of particles.
 * @param u Pointer to the array of u velocity components.
 * @param v Pointer to the array of v velocity components.
 * @param w Pointer to the array of w velocity components.
 * @param histogramCUDAPtr Pointer to the array of 3 velocityHistogramCUDA objects.
 */
__global__ void velocityHistogramKernel(int nop, histogramTypeIn* u, histogramTypeIn* v, histogramTypeIn* w,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pidx >= nop)return;

    const histogramTypeIn uvw[3] = {u[pidx], v[pidx], w[pidx]};
    const histogramTypeIn uv[2] = {uvw[0], uvw[1]};
    const histogramTypeIn vw[2] = {uvw[1], uvw[2]};
    const histogramTypeIn uw[2] = {uvw[0], uvw[2]};

    histogramCUDAPtr[0].addData(uv, 1);
    histogramCUDAPtr[1].addData(vw, 1);
    histogramCUDAPtr[2].addData(uw, 1);

}

__global__ void velocityHistogramKernel(int nop, histogramTypeIn* u, histogramTypeIn* v, histogramTypeIn* w, histogramTypeIn* q,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pidx >= nop)return;

    const histogramTypeIn uvw[3] = {u[pidx], v[pidx], w[pidx]};
    const histogramTypeIn uv[2] = {uvw[0], uvw[1]};
    const histogramTypeIn vw[2] = {uvw[1], uvw[2]};
    const histogramTypeIn uw[2] = {uvw[0], uvw[2]};

    const histogramTypeOut qAbs = abs(q[pidx] * 10e5); // float
    //const int qAbs = 1;

    histogramCUDAPtr[0].addData(uv, qAbs);
    histogramCUDAPtr[1].addData(vw, qAbs);
    histogramCUDAPtr[2].addData(uw, qAbs);

}



__global__ void velocityHistogramKernelOne(int nop, histogramTypeIn* d1, histogramTypeIn* d2, histogramTypeIn* q,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;
    

    extern __shared__ histogramTypeOut sHistogram[];

    histogramTypeIn d1d2[2];
    histogramTypeOut qAbs; // float

    // Initialize shared memory to zero
    for (int i = threadIdx.x; i < histogramCUDAPtr[0].getLogicSize(); i += blockDim.x) {
        sHistogram[i] = 0.0;
    }
    __syncthreads();

    for(int i = pidx; i < nop; i += gridSize){
        
        d1d2[0] = d1[i];
        d1d2[1] = d2[i];

        qAbs = abs(q[i] * 10e5); 

        const auto index = histogramCUDAPtr[0].getIndex(d1d2);
        if(index < 0)continue;
        atomicAdd(&sHistogram[index], qAbs);
    }

    __syncthreads();

    // use one warp to update the histogram
    const auto histogramSize = histogramCUDAPtr[0].getLogicSize();
    auto gHistogram = histogramCUDAPtr[0].getHistogramCUDA();

    if(threadIdx.x < WARP_SIZE){
        for(int i = threadIdx.x; i < histogramSize; i += WARP_SIZE){
            atomicAdd(&gHistogram[i], sHistogram[i]);
        }
    }

}

/**
 * @brief reset and calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin for all 3 histograms
 */
__global__ void resetBinScaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= histogramCUDAPtr->getLogicSize())return;

    
    histogramCUDAPtr[0].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[0].centerOfBin(idx);
    histogramCUDAPtr[1].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[1].centerOfBin(idx);
    histogramCUDAPtr[2].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[2].centerOfBin(idx);

}



} // namespace velocityHistogram







