#ifndef _PARTICLE_ARRAY_CUDA_H_
#define _PARTICLE_ARRAY_CUDA_H_

#include <stdexcept>
#include <iostream>
#include "cudaTypeDef.cuh"
#include "Particle.h"
#include "Particles3D.h"
#include "arrayCUDA.cuh"

class particleArrayCUDA : public arrayCUDA<SpeciesParticle, 32>
{
private:
    ParticleType::Type type;
    uint32_t initialNOP;

public:
    /**
     * @brief create and copy the array from host, bigger size
     * 
     * @param stream the stream used for memory operations
     */ 
    __host__ particleArrayCUDA(Particles3D* p3D, cudaTypeSingle expand = 1.2, cudaStream_t stream = 0): arrayCUDA(p3D->get_pclptr(0), p3D->getNOP(), expand), type(p3D->get_particleType()){
        assignStream(stream);
    }

    __host__ virtual particleArrayCUDA* copyToDevice(){
        particleArrayCUDA* ptr = nullptr;
        cudaErrChk(cudaMalloc((void**)&ptr, sizeof(particleArrayCUDA)));
        cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(particleArrayCUDA), cudaMemcpyDefault, stream));

        cudaErrChk(cudaStreamSynchronize(stream));
        return ptr;
    }


    __host__ __device__ uint32_t getNOP(){ return getNOE(); }
    __host__ __device__ void setInitialNOP(uint32_t numberPcl){ this->initialNOP = numberPcl; }
    __host__ __device__ uint32_t getInitialNOP(){ return initialNOP; }
    __host__ __device__ SpeciesParticle* getpcls(){ return getArray(); }
    __host__ __device__ SpeciesParticle* getpcl(uint32_t index){ return getElement(index); }

    


};



#endif