

#include <iostream>
#include <math.h>
#include <limits.h>
#include "asserts.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Grid3DCU.h"
#include "Field.h"
#include "ipicdefs.h"
#include "TimeTasks.h"
#include "parallel.h"
#include "Particles3D.h"

#include "mic_particles.h"
#include "debug.h"
#include <complex>

#include "cudaTypeDef.cuh"
#include "moverKernel.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "hashedSum.cuh"

using commonType = cudaParticleType;
using namespace std;

__device__ constexpr bool cap_velocity() { return false; }


__device__ void prepareDepartureArray(SpeciesParticle* pcl, 
                                    moverParameter *moverParam,
                                    departureArrayType* departureArray, 
                                    grid3DCUDA* grid, 
                                    hashedSum* hashedSumArray, 
                                    uint32_t pidx);

//* ============================================================================================================================== *//

//! IMM - Implicit moment method - mover (AoS) with a Predictor-Corrector scheme !//
__global__ void moverKernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;
    
    const commonType dto2 = .5 * moverParam->dt,
        qdto2mc = moverParam->qom * dto2 / moverParam->c;

    // copy the particle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;

    if(moverParam->departureArray->getArray()[pidx].dest != 0){ // deleted during the merging
        prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }


    const commonType xorig = pcl->get_x();
    const commonType yorig = pcl->get_y();
    const commonType zorig = pcl->get_z();
    const commonType uorig = pcl->get_u();
    const commonType vorig = pcl->get_v();
    const commonType worig = pcl->get_w();
    commonType xavg = xorig;
    commonType yavg = yorig;
    commonType zavg = zorig;
    commonType uavg, vavg, wavg;
    commonType uavg_old = uorig;
    commonType vavg_old = vorig;
    commonType wavg_old = worig;

    int innter = 0;
    const cudaTypeDouble PC_err_2 = 1E-12;  // square of error tolerance
    cudaTypeDouble currErr = PC_err_2 + 1.; // initialize to a larger value

    // calculate the average velocity iteratively
    while (currErr > PC_err_2 && innter < moverParam->NiterMover)
    {

        // compute weights for field components
        //
        commonType weights[8];
        int cx, cy, cz;
        grid->get_safe_cell_and_weights(xavg, yavg, zavg, cx, cy, cz, weights);

        commonType sampled_field[6];
        for (int i = 0; i < 6; i++)
            sampled_field[i] = 0;
        commonType &Bxl = sampled_field[0];
        commonType &Byl = sampled_field[1];
        commonType &Bzl = sampled_field[2];
        commonType &Exl = sampled_field[3];
        commonType &Eyl = sampled_field[4];
        commonType &Ezl = sampled_field[5];

        // target previous cell
        const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz; // previous cell index

        assert(previousIndex < 24 * (grid->nzn * (grid->nyn - 1) * (grid->nxn - 1)));

        for (int c = 0; c < 8; c++) // grid node
        {
            // 4 from previous and 4 from itself

            for (int i = 0; i < 6; i++) // field items
            {
                sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
            }
        }
        const commonType Omx = qdto2mc * Bxl;
        const commonType Omy = qdto2mc * Byl;
        const commonType Omz = qdto2mc * Bzl;

        // end interpolation
        const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
        const commonType denom = 1.0 / (1.0 + omsq);
        // solve the position equation
        const commonType ut = uorig + qdto2mc * Exl;
        const commonType vt = vorig + qdto2mc * Eyl;
        const commonType wt = worig + qdto2mc * Ezl;
        // const commonType udotb = ut * Bxl + vt * Byl + wt * Bzl;
        const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
        // solve the velocity equation
        uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
        vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
        wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
        // update average position
        xavg = xorig + uavg * dto2;
        yavg = yorig + vavg * dto2;
        zavg = zorig + wavg * dto2;

        innter++;
        currErr = ((uavg_old - uavg) * (uavg_old - uavg) + (vavg_old - vavg) * (vavg_old - vavg) + (wavg_old - wavg) * (wavg_old - wavg)) /
                  (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
        // capture the new velocity for the next iteration
        uavg_old = uavg;
        vavg_old = vavg;
        wavg_old = wavg;

    } // end of iteration

    // update the final position and velocity
    if (cap_velocity()) //used to limit the speed of particles under c
    {
        auto umax = moverParam->umax;
        auto vmax = moverParam->vmax;
        auto wmax = moverParam->wmax;
        auto umin = moverParam->umin;
        auto vmin = moverParam->vmin;
        auto wmin = moverParam->wmin;

        bool cap = (abs(uavg) > umax || abs(vavg) > vmax || abs(wavg) > wmax) ? true : false;
        // we could do something more smooth or sophisticated
        if (cap)
        {
            if (uavg > umax)
                uavg = umax;
            else if (uavg < umin)
                uavg = umin;
            if (vavg > vmax)
                vavg = vmax;
            else if (vavg < vmin)
                vavg = vmin;
            if (wavg > wmax)
                wavg = wmax;
            else if (wavg < wmin)
                wavg = wmin;
        }
    }
    //


    pcl->set_x_u(   xorig + uavg * moverParam->dt,  yorig + vavg * moverParam->dt,  zorig + wavg * moverParam->dt,
                    2.0f * uavg - uorig,            2.0f * vavg - vorig,            2.0f * wavg - worig);



    // prepare the departure array

    prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
    
}

//! IMM - Implicit moment method - mover (AoS with subcycling) with a Predictor-Corrector scheme !//
__global__ void moverSubcyclesKernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    // copy the particle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;

    if(moverParam->departureArray->getArray()[pidx].dest != 0){ // deleted during the merging
        prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }

    // first step: evaluate local B magnitude

    commonType weights[8];

    int cx, cy, cz;
    grid->get_safe_cell_and_weights(pcl->get_x(), pcl->get_y(), pcl->get_z(), cx, cy, cz, weights);
    
    const int previousIndex0 = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz; // previous cell index
    assert(previousIndex0 < 24 * (grid->nzn * (grid->nyn - 1) * (grid->nxn - 1)));

    commonType sampled_field[6];
    for (int i = 0; i < 6; i++)
        sampled_field[i] = 0;  

    for (int c = 0; c < 8; c++) // grid node
    {
        // 4 from previous and 4 from itself

        for (int i = 0; i < 3; i++) // B field items
        {
            sampled_field[i] += weights[c] * fieldForPcls[previousIndex0 * 24 + c * 6 + i];
        }
    }

    // evaluate local B field magnitude
    const commonType B_mag = sqrt(sampled_field[0] * sampled_field[0] + sampled_field[1] * sampled_field[1] + sampled_field[2] * sampled_field[2]);

    // evaluate dt_substep and number of sub cycles

    const commonType dto2 = .5 * moverParam->dt;
    [[maybe_unused]] const commonType qdto2mc = moverParam->qom * dto2 / moverParam->c;

    commonType dt_sub = M_PI * moverParam->c / (4 * fabs(moverParam->qom) * B_mag);
    const int sub_cycles = (int)(moverParam->dt / dt_sub) + 1;
    dt_sub = moverParam->dt / (commonType)(sub_cycles);
    
    const commonType dto2_sub = .5 * dt_sub;
    const commonType qdto2mc_sub = moverParam->qom * dto2_sub / moverParam->c;

    //start subcycling
    for(int cyc_cnt = 0; cyc_cnt < sub_cycles; cyc_cnt++)
    {
        const commonType xorig = pcl->get_x();
        const commonType yorig = pcl->get_y();
        const commonType zorig = pcl->get_z();
        const commonType uorig = pcl->get_u();
        const commonType vorig = pcl->get_v();
        const commonType worig = pcl->get_w();
        commonType xavg = xorig;
        commonType yavg = yorig;
        commonType zavg = zorig;
        commonType uavg, vavg, wavg;
        commonType uavg_old = uorig;
        commonType vavg_old = vorig;
        commonType wavg_old = worig;

        assert( (uorig*uorig + vorig*vorig + worig*worig) < 1 );
        const commonType gamma0 = 1.0 / (sqrt(1.0 - uorig*uorig - vorig*vorig - worig*worig));
        commonType gamma1;

        int innter = 0;
        const cudaTypeDouble PC_err_2 = 1E-12;  // square of error tolerance
        cudaTypeDouble currErr = PC_err_2 + 1.; // initialize to a larger value

        // calculate the average velocity iteratively - predictor corrector
        // uses dt_subcycle
        while (currErr > PC_err_2 && innter < moverParam->NiterMover)
        {

            // compute weights for field components
            grid->get_safe_cell_and_weights(xavg, yavg, zavg, cx, cy, cz, weights);

            for (int i = 0; i < 6; i++)
                sampled_field[i] = 0;

            commonType &Bxl = sampled_field[0];
            commonType &Byl = sampled_field[1];
            commonType &Bzl = sampled_field[2];
            commonType &Exl = sampled_field[3];
            commonType &Eyl = sampled_field[4];
            commonType &Ezl = sampled_field[5];

            // target previous cell
            const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz; // previous cell index
            assert(previousIndex < 24 * (grid->nzn * (grid->nyn - 1) * (grid->nxn - 1)));

            for (int c = 0; c < 8; c++) // grid node
            {
                // 4 from previous and 4 from itself

                for (int i = 0; i < 6; i++) // field items
                {
                    sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
                }
            }
            const commonType Omx = qdto2mc_sub * Bxl;
            const commonType Omy = qdto2mc_sub * Byl;
            const commonType Omz = qdto2mc_sub * Bzl;

            // end interpolation
            const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
            commonType denom = 1.0 / (1.0 + omsq);
            // solve the position equation
            const commonType ut = uorig * gamma0 + qdto2mc_sub * Exl;
            const commonType vt = vorig * gamma0 + qdto2mc_sub * Eyl;
            const commonType wt = worig * gamma0 + qdto2mc_sub * Ezl;

            gamma1 = sqrt(1.0 + ut*ut + vt*vt + wt*wt);
			Bxl /= gamma1;
            Byl /= gamma1;
            Bzl /= gamma1;
            denom /= gamma1;

            // const commonType udotb = ut * Bxl + vt * Byl + wt * Bzl;
            const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
            // solve the velocity equation
            uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
            vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
            wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
            // update average position
            xavg = xorig + uavg * dto2_sub;
            yavg = yorig + vavg * dto2_sub;
            zavg = zorig + wavg * dto2_sub;

            currErr = ((uavg_old - uavg) * (uavg_old - uavg) + (vavg_old - vavg) * (vavg_old - vavg) + (wavg_old - wavg) * (wavg_old - wavg)) /
                        (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
            // capture the new velocity for the next iteration
            uavg_old = uavg;
            vavg_old = vavg;
            wavg_old = wavg;

            innter++;

        } // end of iteration

        // relativistic velocity update
        const commonType ut = uorig * gamma0;
        const commonType vt = vorig * gamma0;
        const commonType wt = worig * gamma0;

        const commonType velt_sq = ut*ut + vt*vt + wt*wt;
        const commonType velavg_sq = uavg*uavg + vavg*vavg + wavg*wavg;
        const commonType velt_velavg = ut*uavg + vt*vavg + wt*wavg;

        const commonType cfa = 1.0 - velavg_sq;
        const commonType cfb = -2.0 * (-velt_velavg + gamma0 * velavg_sq);
        const commonType cfc = -1.0 - gamma0 * gamma0 * velavg_sq + 2.0 * gamma0 * velt_velavg - velt_sq;
        
        const commonType delta_rel = cfb * cfb - 4.0 * cfa * cfc;

         // update velocity
        if (delta_rel < 0.0){

            //cout << "Relativity violated: gamma0=" << gamma0 << ",  vavg_sq=" << vavg_sq;
            pcl->set_x_u(   xorig + uavg * dt_sub,  yorig + vavg * dt_sub,  zorig + wavg * dt_sub,
                (2.0*gamma1)*uavg - uorig*gamma0, (2.0*gamma1)*vavg - vorig*gamma0, (2.0*gamma1)*wavg - worig*gamma0);
        }
        else{
            const commonType gamma1 = ( -cfb + sqrt(delta_rel)) / 2.0 / cfa;
            pcl->set_x_u(   xorig + uavg * dt_sub,  yorig + vavg * dt_sub,  zorig + wavg * dt_sub,
                (1.0 + gamma0/gamma1)*uavg - ut/gamma1, (1.0 + gamma0/gamma1)*vavg - vt/gamma1, (1.0 + gamma0/gamma1)*wavg - wt/gamma1);
        }

    } // end iteration over subcycles
    
    // prepare the departure array

    prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
}

//! select kernel in /src/ipic3d/main/iPIC3Dlib.cu

//! ============================================================================= !//

//! ECSIM - energy conserving semi-implicit method !//
__global__ void ECSIM_velocity_kernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid)
{
    const uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if (pidx >= pclsArray->getNOP()) return;

    // If this particle was deleted/consumed by merging, skip velocity update.
    if (moverParam->departureArray->getArray()[pidx].dest != 0) return;

    // Constants
    const commonType qdto2mc = (commonType)(0.5) * moverParam->dt * moverParam->qom / moverParam->c;

    // Particle handle
    SpeciesParticle* pcl = pclsArray->getpcls() + pidx;

    // Positions/velocities at time n
    const commonType x_n = pcl->get_x();
    const commonType y_n = pcl->get_y();
    const commonType z_n = pcl->get_z();
    const commonType u_n = pcl->get_u();
    const commonType v_n = pcl->get_v();
    const commonType w_n = pcl->get_w();

    // --- Gather E,B at (x_n, y_n, z_n) ---
    commonType weights[8];
    int cx, cy, cz;
    grid->get_safe_cell_and_weights(x_n, y_n, z_n, cx, cy, cz, weights);

    // Flattened field access (must match host packing)
    // NOTE: this assumes 8 nodes × 6 comps (Bx,By,Bz,Ex,Ey,Ez) per cell packed as 24 or 48 values per "previousIndex" stride,
    // exactly like your moverKernel. Keep stride consistent with how fieldForPcls was built.
    commonType sampled_field[6]; // [Bx,By,Bz,Ex,Ey,Ez]
    #pragma unroll
    for (int i = 0; i < 6; ++i) sampled_field[i] = (commonType)0;

    // previous cell index (same convention as moverKernel)
    const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;

    // Accumulate from 8 grid nodes
    #pragma unroll
    for (int c = 0; c < 8; ++c) 
    {
        const commonType wc = weights[c];
        // If your packing is 8 nodes × 6 comps per cell, use stride = 48.
        // If your packing is 4 nodes × 6 comps (24), adjust both the stride and the c-loop accordingly.
        const int base = previousIndex * 24 + c * 6;  // keep consistent with moverKernel’s current indexing
        
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            sampled_field[i] += wc * fieldForPcls[base + i];
    }

    const commonType Bxl = sampled_field[0];
    const commonType Byl = sampled_field[1];
    const commonType Bzl = sampled_field[2];
    const commonType Exl = sampled_field[3];
    const commonType Eyl = sampled_field[4];
    const commonType Ezl = sampled_field[5];

    // --- ECSIM velocity update (no position update) ---
    const commonType u_temp = u_n + qdto2mc * Exl;
    const commonType v_temp = v_n + qdto2mc * Eyl;
    const commonType w_temp = w_n + qdto2mc * Ezl;

    const commonType Omx = qdto2mc * Bxl;
    const commonType Omy = qdto2mc * Byl;
    const commonType Omz = qdto2mc * Bzl;
    const commonType omsq  = Omx*Omx + Omy*Omy + Omz*Omz;
    const commonType denom = (commonType)1 / ((commonType)1 + omsq);
    const commonType udotOm = u_temp * Omx + v_temp * Omy + w_temp * Omz;

    const commonType uavg = (u_temp + (v_temp * Omz - w_temp * Omy + udotOm * Omx)) * denom;
    const commonType vavg = (v_temp + (w_temp * Omx - u_temp * Omz + udotOm * Omy)) * denom;
    const commonType wavg = (w_temp + (u_temp * Omy - v_temp * Omx + udotOm * Omz)) * denom;

    // Velocities at n+1 (ECSIM: update velocities only)
    pcl->set_u((commonType)2 * uavg - u_n);
    pcl->set_v((commonType)2 * vavg - v_n);
    pcl->set_w((commonType)2 * wavg - w_n);
}


/*
! resDivC layout: I used a flat array with index ((ix*nyc)+iy)*nzc + iz. If you already have a device wrapper (e.g., cudaTypeArray3<...>), swap the accessor accordingly.

! Grid accessors: I called grid->getXC/YC/ZC like on CPU. If your device grid3DCUDA stores centers directly (arrays), you may want to inline reads for speed.

! Relativity flag: I assumed a bool relativistic exists in moverParameter. If not, pass a uint8_t or compute gamma unconditionally with Relativistic ? ... : 1.

! Guard cells: The CPU code relies on guard layers so the (ix±1, iy±1, iz±1) accesses are valid. Ensure your device resDivC includes those guards and that ix,iy,iz are in-range for all particles.

! fixPosition(): On GPU, departure handling (prepareDepartureArray) is the standard replacement. If you still need a final “wrap/clip” (e.g., periodic BCs in-place), that should be inside prepareDepartureArray or a dedicated kernel.

*/

__global__ void ECSIM_RelSIM_position_kernel(moverParameter *moverParam, grid3DCUDA *grid)
{
    //* cell-centered R (size = nxc*nyc*nzc)

    const uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if (pidx >= pclsArray->getNOP()) return;

    // If this particle was deleted (e.g., by merging), just finalize departure bookkeeping and exit.
    //TODO: check this
    // if (moverParam->departureArray->getArray()[pidx].dest != 0) return;

    if (moverParam->departureArray->getArray()[pidx].dest != 0) 
    {
        SpeciesParticle *pcl_del = pclsArray->getpcls() + pidx;
        prepareDepartureArray(pcl_del, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }

    // Constants & geometry
    // const commonType dx      = moverParam->dx;
    // const commonType dy      = moverParam->dy;
    // const commonType dz      = moverParam->dz;
    // const commonType inv_dx  = (commonType)1 / dx;
    // const commonType inv_dy  = (commonType)1 / dy;
    // const commonType inv_dz  = (commonType)1 / dz;
    const commonType dt      = moverParam->dt;
    const commonType c       = moverParam->c;
    // const commonType xstart  = moverParam->xstart;
    // const commonType ystart  = moverParam->ystart;
    // const commonType zstart  = moverParam->zstart;
    // const bool Relativistic  = moverParam->Relativistic;   // bool flag carried in moverParam

    // Optional anisotropic correction factors (match CPU defaults = 1)
    const commonType correct_x = (commonType)1.0;
    const commonType correct_y = (commonType)1.0;
    const commonType correct_z = (commonType)1.0;

    // Shortcuts for grid sizes (cell-centered)
    const int nxc = grid->nxc;
    const int nyc = grid->nyc;
    const int nzc = grid->nzc;

    // Particle handle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;

    // State at time n
    const commonType x_n = pcl->get_x();
    const commonType y_n = pcl->get_y();
    const commonType z_n = pcl->get_z();
    const commonType u_n = pcl->get_u();
    const commonType v_n = pcl->get_v();
    const commonType w_n = pcl->get_w();

    // Relativistic gamma at time n (or 1.0 if non-relativistic)
    commonType lorentz_factor = (commonType)1.0;
    // if (Relativistic) 
    // {
    //     const commonType v2 = u_n*u_n + v_n*v_n + w_n*w_n;
    //     lorentz_factor = sqrt((commonType)1.0 + v2/(c*c));
    // }

    // Position used for geometric weights (xavg == x_n in CPU code)
    const commonType xavg = x_n;
    const commonType yavg = y_n;
    const commonType zavg = z_n;

    // Locate host-cell indices (match CPU’s floor and +2 stencil offset)
    // const commonType half_dx = (commonType)0.5 * dx;
    // const commonType half_dy = (commonType)0.5 * dy;
    // const commonType half_dz = (commonType)0.5 * dz;

    // const commonType ixd = floor((xavg - half_dx - xstart) * inv_dx);
    // const commonType iyd = floor((yavg - half_dy - ystart) * inv_dy);
    // const commonType izd = floor((zavg - half_dz - zstart) * inv_dz);

    // int ix = 2 + (int)ixd;
    // int iy = 2 + (int)iyd;
    // int iz = 2 + (int)izd;

    // Helper to fetch R(ix,iy,iz) from flattened resDivC (bounds are assumed valid
    // given the same guard layers as on CPU; add checks if needed in debug)
    // auto R_at = [&](int I, int J, int K) -> commonType 
    // {
    //     const int idx = (I * nyc + J) * nzc + K;
    //     return resDivC[idx];
    // };

    // // ----- X-face differences (RxP, RxM) -----
    // commonType eta0  = yavg - grid->getYC(ix,     iy - 1, iz);
    // commonType zeta0 = zavg - grid->getZC(ix,     iy,     iz - 1);
    // commonType eta1  = grid->getYC(ix, iy, iz) - yavg;
    // commonType zeta1 = grid->getZC(ix, iy, iz) - zavg;

    // commonType invSURF = (commonType)1 / (dy * dz);

    // commonType weight00 = eta0 * zeta0 * invSURF;
    // commonType weight01 = eta0 * zeta1 * invSURF;
    // commonType weight10 = eta1 * zeta0 * invSURF;
    // commonType weight11 = eta1 * zeta1 * invSURF;

    // commonType RxP = (commonType)0;
    // RxP += weight00 * R_at(ix,     iy,     iz);
    // RxP += weight01 * R_at(ix,     iy,     iz - 1);
    // RxP += weight10 * R_at(ix,     iy - 1, iz);
    // RxP += weight11 * R_at(ix,     iy - 1, iz - 1);

    // commonType RxM = (commonType)0;
    // RxM += weight00 * R_at(ix - 1, iy,     iz);
    // RxM += weight01 * R_at(ix - 1, iy,     iz - 1);
    // RxM += weight10 * R_at(ix - 1, iy - 1, iz);
    // RxM += weight11 * R_at(ix - 1, iy - 1, iz - 1);

    // // ----- Y-face differences (RyP, RyM) -----
    // commonType xi0  = xavg - grid->getXC(ix - 1, iy,     iz);
    // zeta0           = zavg - grid->getZC(ix,     iy,     iz - 1);
    // commonType xi1  = grid->getXC(ix, iy, iz) - xavg;
    // zeta1           = grid->getZC(ix, iy, iz) - zavg;

    // invSURF = (commonType)1 / (dx * dz);

    // weight00 = xi0 * zeta0 * invSURF;
    // weight01 = xi0 * zeta1 * invSURF;
    // weight10 = xi1 * zeta0 * invSURF;
    // weight11 = xi1 * zeta1 * invSURF;

    // commonType RyP = (commonType)0;
    // RyP += weight00 * R_at(ix,     iy,     iz);
    // RyP += weight01 * R_at(ix,     iy,     iz - 1);
    // RyP += weight10 * R_at(ix - 1, iy,     iz);
    // RyP += weight11 * R_at(ix - 1, iy,     iz - 1);

    // commonType RyM = (commonType)0;
    // RyM += weight00 * R_at(ix,     iy - 1, iz);
    // RyM += weight01 * R_at(ix,     iy - 1, iz - 1);
    // RyM += weight10 * R_at(ix - 1, iy - 1, iz);
    // RyM += weight11 * R_at(ix - 1, iy - 1, iz - 1);

    // // ----- Z-face differences (RzP, RzM) -----
    // commonType RzP = (commonType)0;
    // commonType RzM = (commonType)0;

    // xi0  = xavg - grid->getXC(ix - 1, iy,     iz);
    // eta0 = yavg - grid->getYC(ix,     iy - 1, iz);
    // xi1  = grid->getXC(ix, iy, iz) - xavg;
    // eta1 = grid->getYC(ix, iy, iz) - yavg;

    // invSURF = (commonType)1 / (dx * dy);

    // weight00 = xi0 * eta0 * invSURF;
    // weight01 = xi0 * eta1 * invSURF;
    // weight10 = xi1 * eta0 * invSURF;
    // weight11 = xi1 * eta1 * invSURF;

    // RzP += weight00 * R_at(ix,     iy,     iz);
    // RzP += weight01 * R_at(ix,     iy - 1, iz);
    // RzP += weight10 * R_at(ix - 1, iy,     iz);
    // RzP += weight11 * R_at(ix - 1, iy - 1, iz);

    // RzM += weight00 * R_at(ix,     iy,     iz - 1);
    // RzM += weight01 * R_at(ix,     iy - 1, iz - 1);
    // RzM += weight10 * R_at(ix - 1, iy,     iz - 1);
    // RzM += weight11 * R_at(ix - 1, iy - 1, iz - 1);

    // // ----- Compute charge-conserving displacements dxp/dyp/dzp -----
    // const commonType quarter = (commonType)0.25;
    // commonType dxp = quarter * (RxP - RxM) * dx;
    // commonType dyp = quarter * (RyP - RyM) * dy;
    // commonType dzp = quarter * (RzP - RzM) * dz;

    // // Limiters (avoid divide-by-zero with small epsilon)
    // const commonType eps = (commonType)1e-10;
    // const commonType limiter = (commonType)1;  // matches CPU

    // // Signed min(|d?|, |v/γ * dt|/limiter) with sign opposite to d?
    // auto signed_limited = [&](commonType d, commonType vdt_over_gamma) -> commonType {
    //     const commonType mag_d   = fabs(d);
    //     const commonType cap     = fabs(vdt_over_gamma) / limiter;
    //     const commonType limited = fmin(mag_d, cap);
    //     const commonType sgn     = -d / (fabs(d) + eps); // −sign(d)
    //     return sgn * limited;
    // };

    // dxp = signed_limited(dxp, u_n/lorentz_factor * dt);
    // dyp = signed_limited(dyp, v_n/lorentz_factor * dt);
    // dzp = signed_limited(dzp, w_n/lorentz_factor * dt);

    // ----- Final position update (ECSIM positions) -----
    const commonType x_new = xavg + (u_n/lorentz_factor) * dt * correct_x; //+ dxp;
    const commonType y_new = yavg + (v_n/lorentz_factor) * dt * correct_y; //+ dyp;
    const commonType z_new = zavg + (w_n/lorentz_factor) * dt * correct_z; //+ dzp;

    pcl->set_x(x_new);
    pcl->set_y(y_new);
    pcl->set_z(z_new);

    // ----- Departure bookkeeping (boundaries / exits) -----
    //TODO: some issues with this -- check
    prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
}


__global__ void RelSIM_velocity_kernel(moverParameter *moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid)
{
    const uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if (pidx >= pclsArray->getNOP()) return;

    SpeciesParticle* pcl = pclsArray->getpcls() + pidx;

    // If particle was deleted/consumed by merging, finalize departure bookkeeping and exit.
    if (moverParam->departureArray->getArray()[pidx].dest != 0) 
    {
        prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }

    // Constants
    const commonType dt  = moverParam->dt;
    const commonType qom = moverParam->qom;
    const commonType c   = moverParam->c;

    // q*dt/(2*m*c)
    const commonType q_dt_2mc = (commonType)0.5 * dt * qom / c;

    // State at time n
    const commonType x_n = pcl->get_x();
    const commonType y_n = pcl->get_y();
    const commonType z_n = pcl->get_z();
    const commonType u_n = pcl->get_u();
    const commonType v_n = pcl->get_v();
    const commonType w_n = pcl->get_w();

    // ----- Gather E,B at (x_n,y_n,z_n) (same pattern as moverKernel) -----
    commonType weights[8];
    int cx, cy, cz;
    grid->get_safe_cell_and_weights(x_n, y_n, z_n, cx, cy, cz, weights);

    commonType sampled_field[6]; // [Bx,By,Bz,Ex,Ey,Ez]
    #pragma unroll
    for (int i=0;i<6;++i) sampled_field[i] = (commonType)0;

    // previous cell index (same convention as moverKernel)
    const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;

    // NOTE: keep stride consistent with how fieldForPcls is packed!
    // If packing = 8 nodes × 6 comps → use stride = 48; if 4×6 → stride = 24 and loop c<4, etc.
    const int stride = 24; // CHANGE TO 48 IF YOUR PACKING IS 8*6 PER CELL
    
    #pragma unroll
    for (int cnode = 0; cnode < 8; ++cnode) 
    {
        const int base = previousIndex * stride + cnode * 6;
        const commonType wc = weights[cnode];
        #pragma unroll
        for (int i = 0; i < 6; ++i) 
        {
            sampled_field[i] = sampled_field[i] + (wc * fieldForPcls[base + i]);
        }
    }

    commonType Bxl = sampled_field[0];
    commonType Byl = sampled_field[1];
    commonType Bzl = sampled_field[2];
    const commonType Exl = sampled_field[3];
    const commonType Eyl = sampled_field[4];
    const commonType Ezl = sampled_field[5];

    // External forces placeholder (set to 0 like CPU)
    const commonType Fxl = (commonType)0;
    const commonType Fyl = (commonType)0;
    const commonType Fzl = (commonType)0;

    commonType uavg, vavg, wavg;

    const uint8_t pusher = moverParam->relativistic_pusher;

    if (pusher == Relativistic_pusher::BORIS)
    {
        //! Boris Pusher

        //* u_temp = u^n + q_dt_2mc*c*E + 0.5*dt*F
        const commonType u_temp = u_n + q_dt_2mc * c * Exl + (commonType)0.5 * dt * Fxl;
        const commonType v_temp = v_n + q_dt_2mc * c * Eyl + (commonType)0.5 * dt * Fyl;
        const commonType w_temp = w_n + q_dt_2mc * c * Ezl + (commonType)0.5 * dt * Fzl;

        const commonType gamma_new = sqrt((commonType)1 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp) / (c*c));

        // Scale B with q_dt_2mc / gamma_new
        Bxl = Bxl * q_dt_2mc / gamma_new;
        Byl = Byl * q_dt_2mc / gamma_new;
        Bzl = Bzl * q_dt_2mc / gamma_new;

        const commonType B2 = Bxl*Bxl + Byl*Byl + Bzl*Bzl;

        // This is the algebraic form used in your CPU code
        const commonType u_new = -(Byl*Byl*u_temp) - (Bzl*Bzl*u_temp) + (Bxl*Byl*v_temp) + (Bxl*Bzl*w_temp) - Byl*w_temp + Bzl*v_temp;
        const commonType v_new = -(Bxl*Bxl*v_temp) - (Bzl*Bzl*v_temp) + (Bxl*Byl*u_temp) + (Byl*Bzl*w_temp) - Bzl*u_temp + Bxl*w_temp;
        const commonType w_new = -(Bxl*Bxl*w_temp) - (Byl*Byl*w_temp) + (Bxl*Bzl*u_temp) + (Byl*Bzl*v_temp) - Bxl*v_temp + Byl*u_temp;

        // Final velocities (n+1)
        uavg = u_temp + u_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Exl + (commonType)0.5 * dt * Fxl;
        vavg = v_temp + v_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Eyl + (commonType)0.5 * dt * Fyl;
        wavg = w_temp + w_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Ezl + (commonType)0.5 * dt * Fzl;
    }
    else if (pusher == Relativistic_pusher::BORIS)
    {
        //! Lapenta_Markidis Pusher

        //TODO PJD: Implement this
        // -------------------- Lapenta–Markidis (placeholder) --------------------
        // The CPU version uses std::complex and a cubic solve; re-implementing this
        // robustly on device needs a small complex helper (or cuComplex) and careful numerics.
        //! Until then, we fall back to the Boris pusher (common practice).
        const commonType u_temp = u_n + q_dt_2mc * c * Exl + (commonType)0.5 * dt * Fxl;
        const commonType v_temp = v_n + q_dt_2mc * c * Eyl + (commonType)0.5 * dt * Fyl;
        const commonType w_temp = w_n + q_dt_2mc * c * Ezl + (commonType)0.5 * dt * Fzl;

        const commonType gamma_new = sqrt((commonType)1 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp) / (c*c));

        Bxl = Bxl * q_dt_2mc / gamma_new;
        Byl = Byl * q_dt_2mc / gamma_new;
        Bzl = Bzl * q_dt_2mc / gamma_new;

        const commonType B2 = Bxl*Bxl + Byl*Byl + Bzl*Bzl;

        const commonType u_new = -(Byl*Byl*u_temp) - (Bzl*Bzl*u_temp) + (Bxl*Byl*v_temp) + (Bxl*Bzl*w_temp) - Byl*w_temp + Bzl*v_temp;
        const commonType v_new = -(Bxl*Bxl*v_temp) - (Bzl*Bzl*v_temp) + (Bxl*Byl*u_temp) + (Byl*Bzl*w_temp) - Bzl*u_temp + Bxl*w_temp;
        const commonType w_new = -(Bxl*Bxl*w_temp) - (Byl*Byl*w_temp) + (Bxl*Bzl*u_temp) + (Byl*Bzl*v_temp) - Bxl*v_temp + Byl*u_temp;

        uavg = u_temp + u_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Exl + (commonType)0.5 * dt * Fxl;
        vavg = v_temp + v_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Eyl + (commonType)0.5 * dt * Fyl;
        wavg = w_temp + w_new * (commonType)2 / ((commonType)1 + B2) + q_dt_2mc * Ezl + (commonType)0.5 * dt * Fzl;
    }
    else 
    {
        printf("Incorrect relativistic pusher! Please choose either 0 for 'Boris' or 1 for 'Lapenta_Markidis'\n");
        assert(0 && "Unknown relativistic particle pusher");
        return;
    }

    // Write back new velocities (n+1)
    pcl->set_u(uavg);
    pcl->set_v(vavg);
    pcl->set_w(wavg);
}

//* ============================================================================================================================== *//

//! Boundary Conditions
__device__ uint32_t deleteAppendOpenBCOutflow(SpeciesParticle* pcl, moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray) 
{
    if (!moverParam->doOpenBC) return 0;

    auto& delBdry = moverParam->deleteBoundary;
    auto& openBdry = moverParam->openBoundary;
    
    for (int side = 0; side < 6; side++) {

        if (!moverParam->applyOpenBC[side]) continue;

        const auto direction = side / 2; // x,y,z
        const auto location = pcl->get_x(direction);
        const bool leftRight = side % 2; // 0: left, 1: right

        // delete boundary
        if( (leftRight == 0 && location < delBdry[side]) || (leftRight == 1 && location > delBdry[side]) ) {
            // delete the particle
            return departureArrayElementType::DELETE;
        }

        // open boundary
        if ( (leftRight == 0 && location < openBdry[side]) || (leftRight == 1 && location > openBdry[side]) ) {
            SpeciesParticle newPcl = *pcl;
            
            // update the position
            newPcl.set_x(direction, newPcl.get_x(direction) + (leftRight==0?-1:1) * openBdry[direction*2]);
            newPcl.fetch_x() += newPcl.get_u() * moverParam->dt;
            newPcl.fetch_y() += newPcl.get_v() * moverParam->dt;
            newPcl.fetch_z() += newPcl.get_w() * moverParam->dt;

            newPcl.set_t(114514.0);

            // if the new particle is still in the domain
            
            if (
                newPcl.get_x() > delBdry[0] && newPcl.get_x() < delBdry[1] &&
                newPcl.get_y() > delBdry[2] && newPcl.get_y() < delBdry[3] &&
                newPcl.get_z() > delBdry[4] && newPcl.get_z() < delBdry[5]
                //newPcl.get_x() > grid->xStart && newPcl.get_x() < grid->xEnd &&
                //newPcl.get_y() > grid->yStart && newPcl.get_y() < grid->yEnd &&
                //newPcl.get_z() > grid->zStart && newPcl.get_z() < grid->zEnd
            ) {
                departureArrayElementType element;
                const auto index = moverParam->pclsArray->getNOP() + atomicAdd(&moverParam->appendCountAtomic, 1);
                // check memory overflow
                if (index >= moverParam->pclsArray->getSize()) {
                    printf("Memory overflow in open boundary outflow\n");
                    //__trap();
                    continue;
                }
                memcpy(moverParam->pclsArray->getpcls() + index, &newPcl, sizeof(SpeciesParticle));
                if(newPcl.get_x() < grid->xStart)
                {
                    element.dest = departureArrayElementType::XLOW;
                }
                else if(newPcl.get_x() > grid->xEnd)
                {
                    element.dest = departureArrayElementType::XHIGH;
                }
                else if(newPcl.get_y() < grid->yStart)
                {
                    element.dest = departureArrayElementType::YLOW;
                }
                else if(newPcl.get_y() > grid->yEnd)
                {
                    element.dest = departureArrayElementType::YHIGH;
                }
                else if(newPcl.get_z() < grid->zStart)
                {
                    element.dest = departureArrayElementType::ZLOW;
                }
                else if(newPcl.get_z() > grid->zEnd)
                {
                    element.dest = departureArrayElementType::ZHIGH;
                }
                else element.dest = departureArrayElementType::STAY;

                if(element.dest != 0){
                    element.hashedId = hashedSumArray[element.dest - 1].add(index);
                }else{
                    element.hashedId = 0;
                }
            
                departureArray->getArray()[index] = element;
            }
        }

    }

    return 0;
}

__device__ uint32_t deleteRepopulateInjection(SpeciesParticle* pcl, moverParameter *moverParam, grid3DCUDA *grid) 
{
    if (!moverParam->doRepopulateInjection) return 0;

    auto& doRepopulateInjectionSide = moverParam->doRepopulateInjectionSide;
    auto& repopulateBoundary = moverParam->repopulateBoundary;

    if (
        (doRepopulateInjectionSide[0] && pcl->get_x() < repopulateBoundary[0]) ||
        (doRepopulateInjectionSide[1] && pcl->get_x() > repopulateBoundary[1]) ||
        (doRepopulateInjectionSide[2] && pcl->get_y() < repopulateBoundary[2]) ||
        (doRepopulateInjectionSide[3] && pcl->get_y() > repopulateBoundary[3]) ||
        (doRepopulateInjectionSide[4] && pcl->get_z() < repopulateBoundary[4]) ||
        (doRepopulateInjectionSide[5] && pcl->get_z() > repopulateBoundary[5])
    ) { // In the repoopulate layers
        return departureArrayElementType::DELETE;
    } 
    else 
    {
        return 0;
    }
}

__device__ uint32_t deleteInsideSphere(SpeciesParticle* pcl, moverParameter *moverParam, grid3DCUDA *grid) 
{    
    if (moverParam->doSphere == 0) return 0;

    if(moverParam->doSphere == 1){ // 3D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = pcl->get_x() - sphereOrigin[0];
        const auto dy = pcl->get_y() - sphereOrigin[1];
        const auto dz = pcl->get_z() - sphereOrigin[2];

        if (dx*dx + dy*dy + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::DELETE;
        }
    } else if(moverParam->doSphere == 2){ // 2D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = pcl->get_x() - sphereOrigin[0];
        const auto dz = pcl->get_z() - sphereOrigin[2];

        if (dx*dx + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::DELETE;
        }
    }

    return 0;
}

//! This function applies all relevant BCs for particles
__device__ void prepareDepartureArray(SpeciesParticle* pcl, moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray, uint32_t pidx)
{
    // Cache the device pointer once
    departureArrayElementType* darr = departureArray->getArray(); // must be device ptr

    if(departureArray->getArray()[pidx].dest != 0) 
    {
        departureArray->getArray()[pidx].hashedId = hashedSumArray[departureArray->getArray()[pidx].dest - 1].add(pidx);
        return;
    }
    
    departureArrayElementType element;

    do 
    {
        // OpenBC_outflow
        element.dest = deleteAppendOpenBCOutflow(pcl, moverParam, departureArray, grid, hashedSumArray);
        if(element.dest != 0)break;

        // INJECT
        element.dest = deleteRepopulateInjection(pcl, moverParam, grid);
        if(element.dest != 0)break;

        // sphere
        element.dest = deleteInsideSphere(pcl, moverParam, grid);
        if(element.dest != 0)break;

        // Exiting
        if(pcl->get_x() < grid->xStart)
        {
            element.dest = departureArrayElementType::XLOW;
        }
        else if(pcl->get_x() > grid->xEnd)
        {
            element.dest = departureArrayElementType::XHIGH;
        }
        else if(pcl->get_y() < grid->yStart)
        {
            element.dest = departureArrayElementType::YLOW;
        }
        else if(pcl->get_y() > grid->yEnd)
        {
            element.dest = departureArrayElementType::YHIGH;
        }
        else if(pcl->get_z() < grid->zStart)
        {
            element.dest = departureArrayElementType::ZLOW;
        }
        else if(pcl->get_z() > grid->zEnd)
        {
            element.dest = departureArrayElementType::ZHIGH;
        }
        else element.dest = departureArrayElementType::STAY;

    }
    while (0);

    if(element.dest != 0)
    {
        element.hashedId = hashedSumArray[element.dest - 1].add(pidx);
    }
    else
    {
        element.hashedId = 0;
    }

    departureArray->getArray()[pidx] = element;
}

/**
 * @brief Update the number of particles after the mover kernel, for OBC. Reset OBC
 * @details This should be only called after the moverKernel, if OBC is used. Only one thread
 */
__global__ void updatePclNumAfterMoverKernel(moverParameter *moverParam) 
{
    if (threadIdx.x != 0) return;

    auto pclsArray = moverParam->pclsArray;

    // update the number of particles
    if (moverParam->doOpenBC == true)
    pclsArray->setNOE(pclsArray->getNOP() + moverParam->appendCountAtomic);

    moverParam->appendCountAtomic = 0;
}