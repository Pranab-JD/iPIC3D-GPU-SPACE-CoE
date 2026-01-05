

#include "EMfields3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "CG.h"
#include "GMRES.h"
#include "Particles3Dcomm.h"
#include "Moments.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "debug.h"
#include "string.h"
#include "mic_particles.h"
#include "TimeTasks.h"
#include "ipicmath.h" // for roundup_to_multiple
#include "Alloc.h"
#include "asserts.h"
#include "Particles3D.h"
#include "ECSIM_Moments.h"

#include "cudaTypeDef.cuh"
#include "momentKernel.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

using commonType = cudaTypeDouble; // calculation type

struct int3u { int x,y,z; };

//? Neighbouring Nodes
__device__ __constant__ int NeNoX[14] = { 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,  1,  1, -1, 1 };
__device__ __constant__ int NeNoY[14] = { 0, 0, 1, 0, 1,-1, 0, 0, 1, 1, -1,  1,  1, 1 };
__device__ __constant__ int NeNoZ[14] = { 0, 0, 0, 1, 0, 0, 1,-1, 1,-1,  1, -1,  1, 1 };

//! =================================== IMM Moments =================================== !//

__global__ void momentKernelStayed(momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments)
{

    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gridSize = blockDim.x * gridDim.x;
    auto pclsArray = momentParam->pclsArray;
    const uint totPcl = pclsArray->getNOP();

    for(uint pidx = tidx; pidx < totPcl; pidx += gridSize )
    {
        if(momentParam->departureArray->getArray()[pidx].dest != 0)
            continue; // return the exiting particles, which are out of current domian

        // can be const
        const commonType& inv_dx = grid->invdx;
        const commonType& inv_dy = grid->invdy;
        const commonType& inv_dz = grid->invdz;
        const int& nxn = grid->nxn; // nxn
        const int& nyn = grid->nyn;
        const int& nzn = grid->nzn;
        const commonType& xstart = grid->xStart; // x start
        const commonType& ystart = grid->yStart;
        const commonType& zstart = grid->zStart;
        

        const SpeciesParticle &pcl = pclsArray->getpcls()[pidx];
        // compute the quadratic moments of velocity
        const commonType ui = pcl.get_u();
        const commonType vi = pcl.get_v();
        const commonType wi = pcl.get_w();
        const commonType uui = ui * ui;
        const commonType uvi = ui * vi;
        const commonType uwi = ui * wi;
        const commonType vvi = vi * vi;
        const commonType vwi = vi * wi;
        const commonType wwi = wi * wi;
        commonType velmoments[10];
        velmoments[0] = 1.; // charge density
        velmoments[1] = ui; // momentum density
        velmoments[2] = vi;
        velmoments[3] = wi;
        velmoments[4] = uui; // second time momentum
        velmoments[5] = uvi;
        velmoments[6] = uwi;
        velmoments[7] = vvi;
        velmoments[8] = vwi;
        velmoments[9] = wwi;

        //
        // compute the weights to distribute the moments
        //
        const int ix = 2 + int(floor((pcl.get_x() - xstart) * inv_dx));
        const int iy = 2 + int(floor((pcl.get_y() - ystart) * inv_dy));
        const int iz = 2 + int(floor((pcl.get_z() - zstart) * inv_dz));
        const commonType xi0 = pcl.get_x() - grid->getXN(ix-1); // calculate here
        const commonType eta0 = pcl.get_y() - grid->getYN(iy - 1);
        const commonType zeta0 = pcl.get_z() - grid->getZN(iz - 1);
        const commonType xi1 = grid->getXN(ix) - pcl.get_x();
        const commonType eta1 = grid->getYN(iy) - pcl.get_y();
        const commonType zeta1 = grid->getZN(iz) - pcl.get_z();
        const commonType qi = pcl.get_q();
        const commonType invVOLqi = grid->invVOL * qi;
        const commonType weight0 = invVOLqi * xi0;
        const commonType weight1 = invVOLqi * xi1;
        const commonType weight00 = weight0 * eta0;
        const commonType weight01 = weight0 * eta1;
        const commonType weight10 = weight1 * eta0;
        const commonType weight11 = weight1 * eta1;
        commonType weights[8]; // put the invVOL here
        weights[0] = weight00 * zeta0 * grid->invVOL; // weight000
        weights[1] = weight00 * zeta1 * grid->invVOL; // weight001
        weights[2] = weight01 * zeta0 * grid->invVOL; // weight010
        weights[3] = weight01 * zeta1 * grid->invVOL; // weight011
        weights[4] = weight10 * zeta0 * grid->invVOL; // weight100
        weights[5] = weight10 * zeta1 * grid->invVOL; // weight101
        weights[6] = weight11 * zeta0 * grid->invVOL; // weight110
        weights[7] = weight11 * zeta1 * grid->invVOL; // weight111


        uint64_t posIndex[8];
        posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
        posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
        posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
        posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
        posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
        posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
        posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
        posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
        
        uint64_t oneDensity = nxn * nyn * nzn;
        for (int m = 0; m < 10; m++)    // 10 densities
        for (int c = 0; c < 8; c++)     // 8 grid nodes
        {
            atomicAdd(&moments[oneDensity*m + posIndex[c]], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
        }
    }
}

__global__ void momentKernelNew(momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle)
{

    uint pidx = stayedParticle + blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = momentParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    // can be shared
    const commonType inv_dx = 1.0 / grid->dx;
    const commonType inv_dy = 1.0 / grid->dy;
    const commonType inv_dz = 1.0 / grid->dz;
    const int nxn = grid->nxn; // nxn
    const int nyn = grid->nyn;
    const int nzn = grid->nzn;
    const commonType xstart = grid->xStart; // x start
    const commonType ystart = grid->yStart;
    const commonType zstart = grid->zStart;
    

    const SpeciesParticle &pcl = pclsArray->getpcls()[pidx];
    // compute the quadratic moments of velocity
    const commonType ui = pcl.get_u();
    const commonType vi = pcl.get_v();
    const commonType wi = pcl.get_w();
    const commonType uui = ui * ui;
    const commonType uvi = ui * vi;
    const commonType uwi = ui * wi;
    const commonType vvi = vi * vi;
    const commonType vwi = vi * wi;
    const commonType wwi = wi * wi;
    commonType velmoments[10];
    velmoments[0] = 1.; // charge density
    velmoments[1] = ui; // momentum density
    velmoments[2] = vi;
    velmoments[3] = wi;
    velmoments[4] = uui; // second time momentum
    velmoments[5] = uvi;
    velmoments[6] = uwi;
    velmoments[7] = vvi;
    velmoments[8] = vwi;
    velmoments[9] = wwi;

    //
    // compute the weights to distribute the moments
    //
    const int ix = 2 + int(floor((pcl.get_x() - xstart) * inv_dx));
    const int iy = 2 + int(floor((pcl.get_y() - ystart) * inv_dy));
    const int iz = 2 + int(floor((pcl.get_z() - zstart) * inv_dz));
    const commonType xi0 = pcl.get_x() - grid->getXN(ix-1); // calculate here
    const commonType eta0 = pcl.get_y() - grid->getYN(iy - 1);
    const commonType zeta0 = pcl.get_z() - grid->getZN(iz - 1);
    const commonType xi1 = grid->getXN(ix) - pcl.get_x();
    const commonType eta1 = grid->getYN(iy) - pcl.get_y();
    const commonType zeta1 = grid->getZN(iz) - pcl.get_z();
    const commonType qi = pcl.get_q();
    const commonType invVOLqi = grid->invVOL * qi;
    const commonType weight0 = invVOLqi * xi0;
    const commonType weight1 = invVOLqi * xi1;
    const commonType weight00 = weight0 * eta0;
    const commonType weight01 = weight0 * eta1;
    const commonType weight10 = weight1 * eta0;
    const commonType weight11 = weight1 * eta1;
    commonType weights[8]; // put the invVOL here
    weights[0] = weight00 * zeta0 * grid->invVOL; // weight000
    weights[1] = weight00 * zeta1 * grid->invVOL; // weight001
    weights[2] = weight01 * zeta0 * grid->invVOL; // weight010
    weights[3] = weight01 * zeta1 * grid->invVOL; // weight011
    weights[4] = weight10 * zeta0 * grid->invVOL; // weight100
    weights[5] = weight10 * zeta1 * grid->invVOL; // weight101
    weights[6] = weight11 * zeta0 * grid->invVOL; // weight110
    weights[7] = weight11 * zeta1 * grid->invVOL; // weight111


    uint64_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
    
    uint64_t oneDensity = nxn * nyn * nzn;
    for (int m = 0; m < 10; m++)    // 10 densities
    for (int c = 0; c < 8; c++)     // 8 grid nodes
    {
        atomicAdd(&moments[oneDensity*m + posIndex[c]], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
    }
}


//! =============================== ECSIM/RelSIM Moments =============================== !//

__device__ inline void exact_mass_matrix(cudaTypeArray1<cudaMomentType> moments,
                                        int ix,int iy,int iz,
                                        double q, double q_dt_2mc,
                                        const double weights[8],
                                        double a00, double a01, double a02,
                                        double a10, double a11, double a12,
                                        double a20, double a21, double a22,
                                        int nxn,int nyn,int nzn)
{
    using namespace moments130;

    const uint64_t oneDensity = (uint64_t)nxn * (uint64_t)nyn * (uint64_t)nzn;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) 
            {
                const int ni = ix - i, nj = iy - j, nk = iz - k;
                const int idx1 = i*4 + j*2 + k;                     // 0..7 for (i,j,k)
                const uint64_t nodeIdx = toOneDimIndex((uint64_t)nxn,(uint64_t)nyn,(uint64_t)nzn, (uint64_t)ni,(uint64_t)nj,(uint64_t)nk);

                #pragma unroll
                for (int ind = 0; ind < NUM_NEIGHBORS; ++ind)
                {
                    const int n2i = ni + NeNoX[ind];
                    const int n2j = nj + NeNoY[ind];
                    const int n2k = nk + NeNoZ[ind];
                    
                    const int i2 = ix - n2i, j2 = iy - n2j, k2 = iz - n2k;

                    //? (i2>=0 && i2<2 && j2>=0 && j2<2 && k2>=0 && k2<2)
                    if ( (unsigned)i2 < 2u && (unsigned)j2 < 2u && (unsigned)k2 < 2u )
                    {
                        const int idx2 = i2*4 + j2*2 + k2; // 0..7 for (i2,j2,k2)
                        const double qww = q * q_dt_2mc * weights[idx1] * weights[idx2];
                        const uint64_t base = 4u + (uint64_t)(ind * 9);

                        //? Deposit 3x3 mass block into moment slots 4..12 (row-major: r*3+c)
                        // CPU mapping: value[r][c] = alpha[c][r] * qww
                        // We deposit per-neighbor into channels BASE+ind*9 + (r*3+c)
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 0, 0), oneDensity) + nodeIdx], (cudaMomentType)(a00 * qww)); // Mxx[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 0, 1), oneDensity) + nodeIdx], (cudaMomentType)(a01 * qww)); // Mxy[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 0, 2), oneDensity) + nodeIdx], (cudaMomentType)(a02 * qww)); // Mxz[ind]

                        atomicAdd(&moments[chan_offset(mm_channel(ind, 1, 0), oneDensity) + nodeIdx], (cudaMomentType)(a10 * qww)); // Myx[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 1, 1), oneDensity) + nodeIdx], (cudaMomentType)(a11 * qww)); // Myy[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 1, 2), oneDensity) + nodeIdx], (cudaMomentType)(a12 * qww)); // Myz[ind]

                        atomicAdd(&moments[chan_offset(mm_channel(ind, 2, 0), oneDensity) + nodeIdx], (cudaMomentType)(a20 * qww)); // Mzx[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 2, 1), oneDensity) + nodeIdx], (cudaMomentType)(a21 * qww)); // Mzy[ind]
                        atomicAdd(&moments[chan_offset(mm_channel(ind, 2, 2), oneDensity) + nodeIdx], (cudaMomentType)(a22 * qww)); // Mzz[ind]
                    }
                }
            }
}

__global__ void ECSIM_RelSIM_Moments_PostExchange(momentParameter* momentParam,
                                                  grid3DCUDA* grid,
                                                  cudaTypeArray1<cudaFieldType> fieldForPcls,
                                                  cudaTypeArray1<cudaMomentType> moments,
                                                  int stayedParticle)
{
    //? Skip over all particles that remained in the same MPI domain
    uint pidx = stayedParticle + blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = momentParam->pclsArray;
    if(pidx >= pclsArray->getNOP()) return;

    const int nxn = grid->nxn, nyn = grid->nyn, nzn = grid->nzn;
    const uint64_t oneDensity = (uint64_t)nxn * (uint64_t)nyn * (uint64_t)nzn;

    const SpeciesParticle& pcl = pclsArray->getpcls()[pidx];
    const commonType x = pcl.get_x(), y = pcl.get_y(), z = pcl.get_z();
    const commonType u = pcl.get_u(), v = pcl.get_v(), w = pcl.get_w();
    const commonType q = pcl.get_q();

    commonType weights[8];
    int cx, cy, cz;
    grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);

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
    const int previousIndex = (cx * (nyn - 1) + cy) * nzn + cz; // previous cell index
    assert(previousIndex < 24 * (nzn * (nyn - 1) * (nxn - 1)));

    for (int c = 0; c < 8; c++) // grid node
    {
        // 4 from previous and 4 from itself
        for (int i = 0; i < 6; i++) // field items
        {
            sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
        }
    }

    // ---------------- Rotation matrix alpha (relativistic-aware) ----------------
    commonType lorentz_factor = (commonType)1.0;

    //     if (momentParam->isRelativistic)
    //     {
    //         if (momentParam->relPusher == 0) // Boris
    //         {
    //             const double u_temp = u_n + q_dt_2mc * c * Exl + 0.5 * dt * Fxl;
    //             const double v_temp = v_n + q_dt_2mc * c * Eyl + 0.5 * dt * Fyl;
    //             const double w_temp = w_n + q_dt_2mc * c * Ezl + 0.5 * dt * Fzl;
    //             lorentz_factor = sqrt(1.0 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp) / (c*c));
    //         }
    //         else if (momentParam->relPusher == 1) // Lapenta_Markidis
    //         {
    //             const double lorentz_factor_old = sqrt(1.0 + (u_n*u_n + v_n*v_n + w_n*w_n) / (c*c));
    //             double beta_x = q_dt_2mc * Bxl;
    //             double beta_y = q_dt_2mc * Byl;
    //             double beta_z = q_dt_2mc * Bzl;
    //             const double B_squared = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

    //             double eps_x = q_dt_2mc * Exl + 0.5 * dt * Fxl;
    //             double eps_y = q_dt_2mc * Eyl + 0.5 * dt * Fyl;
    //             double eps_z = q_dt_2mc * Ezl + 0.5 * dt * Fzl;

    //             const double u_prime = u_n + eps_x;
    //             const double v_prime = v_n + eps_y;
    //             const double w_prime = w_n + eps_z;

    //             const double u_dot_eps  = u_prime*eps_x + v_prime*eps_y + w_prime*eps_z;
    //             const double beta_dot_e = beta_x*eps_x + beta_y*eps_y + beta_z*eps_z;
    //             const double u_dot_beta = u_prime*beta_x + v_prime*beta_y + w_prime*beta_z;

    //             const double u_cross_beta_x =  v_prime*beta_z - w_prime*beta_y;
    //             const double v_cross_beta_y = -u_prime*beta_z + w_prime*beta_x;
    //             const double w_cross_beta_z =  u_prime*beta_y - v_prime*beta_x;

    //             const double aa = u_dot_eps - B_squared;
    //             const double bb = u_cross_beta_x*eps_x + v_cross_beta_y*eps_y + w_cross_beta_z*eps_z + lorentz_factor_old * B_squared;
    //             const double cc = u_dot_beta * beta_dot_e;

    //             const double AA = 2.*aa/3. + lorentz_factor_old*lorentz_factor_old/4.;
    //             const double BB = 4.*aa*lorentz_factor_old + 8.*bb + lorentz_factor_old*lorentz_factor_old*lorentz_factor_old;
    //             const double DD = aa*aa - 3.*bb*lorentz_factor_old - 12.*cc;
    //             const double FF = -2.*aa*aa*aa + 9.*aa*bb*lorentz_factor_old - 72.*aa*cc + 27.*bb*bb - 27.*cc*lorentz_factor_old*lorentz_factor_old;

    //             // Device-friendly cubic handling: crude safe fallback → use old gamma if numerics get tricky
    //             double disc = FF*FF - 4.*DD*DD*DD;
    //             double lorentz_factor_bar = lorentz_factor_old; // fallback

    //             if (isfinite(disc)) {
    //                 // Very lightweight approximation: use positive real branch only
    //                 double rootDisc = (disc >= 0.0) ? sqrt(disc) : 0.0;
    //                 double EE = 0.0;
    //                 double half = 0.5 * (FF + rootDisc);
    //                 if (half >= 0.0) EE = pow(half, 1.0/3.0);
    //                 else             EE = -pow(-half, 1.0/3.0);
    //                 double CCc = (fabs(EE) > 1e-20) ? (DD/(EE)/3.0 + EE/3.0) : 0.0;
    //                 double sq1 = AA + CCc;
    //                 if (sq1 > 0.0) {
    //                     double s1 = sqrt(sq1);
    //                     double sq2 = 2.*AA + BB/(4.*s1) - CCc;
    //                     if (sq2 > 0.0)
    //                         lorentz_factor_bar = lorentz_factor_old/4. + 0.5*sqrt(sq2) + 0.5*s1;
    //                 }
    //             }
    //             lorentz_factor = lorentz_factor_bar;
    //         }
    //         else {
    //             // Unknown pusher → default to Boris gamma for robustness
    //             const double u_temp = u_n + q_dt_2mc * c * Exl + 0.5 * dt * Fxl;
    //             const double v_temp = v_n + q_dt_2mc * c * Eyl + 0.5 * dt * Fyl;
    //             const double w_temp = w_n + q_dt_2mc * c * Ezl + 0.5 * dt * Fzl;
    //             lorentz_factor = sqrt(1.0 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp) / (c*c));
    //         }
    //     }

    const commonType q_dt_2mc = (commonType)0.5 * momentParam->dt * momentParam->qom / momentParam->c;
    const commonType Omx = (q_dt_2mc / lorentz_factor) * Bxl;
    const commonType Omy = (q_dt_2mc / lorentz_factor) * Byl;
    const commonType Omz = (q_dt_2mc / lorentz_factor) * Bzl;
    const commonType omsq = Omx*Omx + Omy*Omy + Omz*Omz;
    const commonType denom = (commonType)1.0 / ((commonType)1.0 + omsq);

    // alpha - rotation matrix
    commonType a00 = ((commonType)1.0 + Omx*Omx) * denom;
    commonType a01 = (Omz + Omx*Omy)             * denom;
    commonType a02 = (-Omy + Omx*Omz)            * denom;

    commonType a10 = (-Omz + Omx*Omy)            * denom;
    commonType a11 = ((commonType)1.0 + Omy*Omy) * denom;
    commonType a12 = (Omx + Omy*Omz)             * denom;

    commonType a20 = (Omy + Omx*Omz)             * denom;
    commonType a21 = (-Omx + Omy*Omz)            * denom;
    commonType a22 = ((commonType)1.0 + Omz*Omz) * denom;

    // q * alpha * [u; v; w]
    const commonType qau = q * (a00*u + a01*v + a02*w);
    const commonType qav = q * (a10*u + a11*v + a12*w);
    const commonType qaw = q * (a20*u + a21*v + a22*w);

    //* Linearised indices (as all arrays on GPU are contiguous arrays)
    const int ix = cx + 1, iy = cy + 1, iz = cz + 1;

    // node linear indices for the 8 corners
    // uint64_t posIndex[8];
    // posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz  );
    // posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz-1);
    // posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz  );
    // posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz-1);
    // posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz  );
    // posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz-1);
    // posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz  );
    // posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);

    //* ------------------------------------------------------------------- *//

    commonType temp_rho[8];
    commonType temp_jx[8];
    // commonType temp_jy[8];
    // commonType temp_jz[8];

    for (int ii = 0; ii < 8; ++ii) 
    {
        temp_rho[ii] = q   * weights[ii];
        temp_jx[ii]  = qau * weights[ii];
        // temp_jy[ii]  = qav * weights[ii];
        // temp_jz[ii]  = qaw * weights[ii];
    }

    // Print only a few particles to avoid flooding
    if (pidx == 0) 
    {
        // printf("Particle velocity %f %f %f\n", u, v, w);
        // printf("Particle position %f %f %f", x, y, z);
        // printf("\n\n");
        // printf("qau, qav, qaw %e %e %e\n\n", qau, qav, qaw);

        // printf("Rho temp\n");
        // for (int ii = 0; ii < 8; ++ii) 
        //     printf("%e   ", temp_rho[ii]);
        // printf("\n\n");

        // printf("Jx temp\n");
        // for (int ii = 0; ii < 8; ++ii) 
        //     printf("%e   ", temp_jx[ii]);
        // printf("\n\n");

        // printf("GPU pidx=%u JZ : ", pidx);
        // for (int ii = 0; ii < 8; ++ii) printf("%e ", temp_jz[ii]);
        // printf("\n\n");
    }

    //* Deposit rho and implicit current (first 4 moments)
    // for (int cc = 0; cc < 8; ++cc) 
    // {
    //     const commonType wc = weights[cc];
    //     const uint64_t idx = posIndex[cc];

    //     atomicAdd(&moments[chan_offset(moments130::CH_RHO, oneDensity) + idx], (cudaMomentType)(q   * wc));
    //     atomicAdd(&moments[chan_offset(moments130::CH_JX,  oneDensity) + idx], (cudaMomentType)(qau * wc));
    //     atomicAdd(&moments[chan_offset(moments130::CH_JY,  oneDensity) + idx], (cudaMomentType)(qav * wc));
    //     atomicAdd(&moments[chan_offset(moments130::CH_JZ,  oneDensity) + idx], (cudaMomentType)(qaw * wc));
    // }

    #pragma unroll
    for (int i = 0; i < 2; ++i) 
    {
        #pragma unroll
        for (int j = 0; j < 2; ++j) 
        {
            #pragma unroll
            for (int k = 0; k < 2; ++k) 
            {
                const int cc = i * 4 + j * 2 + k;     // MUST match get_weights ordering
                const commonType wc = weights[cc];

                // CPU: rhons[X-i][Y-j][Z-k]  (and same for J*)
                const uint64_t node = toOneDimIndex((uint32_t)nxn, (uint32_t)nyn, (uint32_t)nzn,
                                    (uint32_t)(ix - i), (uint32_t)(iy - j), (uint32_t)(iz - k));

                atomicAdd(&moments[chan_offset(moments130::CH_RHO, oneDensity) + node], (cudaMomentType)(q   * wc));
                atomicAdd(&moments[chan_offset(moments130::CH_JX,  oneDensity) + node], (cudaMomentType)(qau * wc));
                atomicAdd(&moments[chan_offset(moments130::CH_JY,  oneDensity) + node], (cudaMomentType)(qav * wc));
                atomicAdd(&moments[chan_offset(moments130::CH_JZ,  oneDensity) + node], (cudaMomentType)(qaw * wc));
            }
        }
    }

    //? Compute exact Mass Matrix
    exact_mass_matrix(moments, ix, iy, iz, q, q_dt_2mc, weights, a00, a01, a02, a10, a11, a12, a20, a21, a22, nxn, nyn, nzn);
}

__global__ void ECSIM_RelSIM_Moments_PreExchange(momentParameter* momentParam,
                                                 grid3DCUDA* grid,
                                                 cudaTypeArray1<cudaFieldType> fieldForPcls,
                                                 cudaTypeArray1<cudaMomentType> moments)
{
    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gridSize = blockDim.x * gridDim.x;
    auto pclsArray = momentParam->pclsArray;
    const uint num_particles = pclsArray->getNOP();                                 //* Total number of particles

    const int nxn = grid->nxn, nyn = grid->nyn, nzn = grid->nzn;
    const uint64_t oneDensity = (uint64_t)nxn * (uint64_t)nyn * (uint64_t)nzn;

    for(uint pidx = tidx; pidx < num_particles; pidx += gridSize)
    {
        if(momentParam->departureArray->getArray()[pidx].dest != 0) continue;       //* Skip over particles that leave their MPI domain

        //? Get particle parameters
        const SpeciesParticle& pcl = pclsArray->getpcls()[pidx];
        const commonType x = pcl.get_x(), y = pcl.get_y(), z = pcl.get_z();
        const commonType u = pcl.get_u(), v = pcl.get_v(), w = pcl.get_w();
        const commonType q = pcl.get_q();

        commonType weights[8];
        int cx, cy, cz;
        grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);
        
        // previous cell index used by your packed field buffer
        const int prevIndex = (cx * (nyn - 1) + cy) * nzn + cz;
        
        // we only need Bx,By,Bz (3 values) from each of 8 nodes
        commonType Bxl = 0, Byl = 0, Bzl = 0, Exl = 0, Eyl = 0, Ezl = 0;

        // fieldForPcls layout: for each node c in {0..7}, the 6-tuple is {Bx,By,Bz,Ex,Ey,Ez}
        // and the 8 nodes per cell make 24 entries per cell (8*3 for B + 8*3 for E).
        // index = prevIndex*24 + c*6 + i
        for (int icx = 0; icx < 8; ++icx) 
        {
            const int base = prevIndex * 24 + icx * 6;      // {Bx,By,Bz,Ex,Ey,Ez}
            const commonType wc = weights[icx];
            Bxl += wc * fieldForPcls[base + 0];
            Byl += wc * fieldForPcls[base + 1];
            Bzl += wc * fieldForPcls[base + 2];
        }

        //* ------------------------------------------------------------------- *//

        const commonType q_dt_2mc = (commonType)0.5 * momentParam->dt * momentParam->qom / momentParam->c;
        const commonType Omx = q_dt_2mc * Bxl;
        const commonType Omy = q_dt_2mc * Byl;
        const commonType Omz = q_dt_2mc * Bzl;
        const commonType omsq = Omx*Omx + Omy*Omy + Omz*Omz;
        const commonType denom = (commonType)1.0 / ((commonType)1.0 + omsq);

        // alpha - rotation matrix
        commonType a00 = ((commonType)1.0 + Omx*Omx) * denom;
        commonType a01 = (Omz + Omx*Omy)             * denom;
        commonType a02 = (-Omy + Omx*Omz)            * denom;

        commonType a10 = (-Omz + Omx*Omy)            * denom;
        commonType a11 = ((commonType)1.0 + Omy*Omy) * denom;
        commonType a12 = (Omx + Omy*Omz)             * denom;

        commonType a20 = (Omy + Omx*Omz)             * denom;
        commonType a21 = (-Omx + Omy*Omz)            * denom;
        commonType a22 = ((commonType)1.0 + Omz*Omz) * denom;

        // q * alpha * [u; v; w]
        const commonType qau = q * (a00*u + a01*v + a02*w);
        const commonType qav = q * (a10*u + a11*v + a12*w);
        const commonType qaw = q * (a20*u + a21*v + a22*w);

        //* ------------------------------------------------------------------- *//

        // //? Map the 3D node index (i,j,k) into a flat index - node indices for CIC deposition
        // const int ix = 2 + int(floor((x - xstart) * grid->invdx));
        // const int iy = 2 + int(floor((y - ystart) * grid->invdy));
        // const int iz = 2 + int(floor((z - zstart) * grid->invdz));

        //* Linearised indices (as all arrays on GPU are contiguous arrays)
        const int ix = cx + 1, iy = cy + 1, iz = cz + 1;

        uint64_t posIndex[8];
        posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz  );
        posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz-1);
        posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz  );
        posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz-1);
        posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz  );
        posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz-1);
        posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz  );
        posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);

        //* ------------------------------------------------------------------- *//

        //* Deposit rho and implicit current (first 4 moments)
        for (int cc = 0; cc < 8; ++cc) 
        {
            const commonType wc = weights[cc];
            const uint64_t idx = posIndex[cc];

            atomicAdd(&moments[chan_offset(moments130::CH_RHO, oneDensity) + idx], (cudaMomentType)(q   * wc));
            atomicAdd(&moments[chan_offset(moments130::CH_JX,  oneDensity) + idx], (cudaMomentType)(qau * wc));
            atomicAdd(&moments[chan_offset(moments130::CH_JY,  oneDensity) + idx], (cudaMomentType)(qav * wc));
            atomicAdd(&moments[chan_offset(moments130::CH_JZ,  oneDensity) + idx], (cudaMomentType)(qaw * wc));
        }

        //? Compute exact Mass Matrix
        exact_mass_matrix(moments, ix, iy, iz, q, q_dt_2mc, weights, a00, a01, a02, a10, a11, a12, a20, a21, a22, nxn, nyn, nzn);
    }
}