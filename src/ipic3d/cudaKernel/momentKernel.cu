

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

//TODO: check all differences between this and following function
//TODO: for loop???

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


        uint32_t posIndex[8];
        posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
        posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
        posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
        posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
        posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
        posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
        posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
        posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
        
        uint32_t oneDensity = nxn * nyn * nzn;
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


    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
    
    uint32_t oneDensity = nxn * nyn * nzn;
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
    uint32_t oneDensity = nxn * nyn * nzn;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) 
            {
                const int ni = ix - i, nj = iy - j, nk = iz - k;
                const int idx1 = i*4 + j*2 + k;                     // 0..7 for (i,j,k)

                for (int n_node=0;n_node<14;++n_node) 
                {
                    const int n2i = ni + NeNoX[n_node];
                    const int n2j = nj + NeNoY[n_node];
                    const int n2k = nk + NeNoZ[n_node];
                    
                    const int i2 = ix - n2i, j2 = iy - n2j, k2 = iz - n2k;
                    
                    // if (i2>=0 && i2<2 && j2>=0 && j2<2 && k2>=0 && k2<2) 
                    // {
                    //     const int idx2 = i2*4 + j2*2 + k2;          // 0..7 for (i,j,k)
                    //     const double qww = q * q_dt_2mc * weights[idx1] * weights[idx2];

                    //     const uint32_t nodeIdx = toOneDimIndex((uint32_t)nxn,(uint32_t)nyn,(uint32_t)nzn, (uint32_t)ni,(uint32_t)nj,(uint32_t)nk);
                        
                    //     #pragma unroll
                    //     for (int r=0;r<3;++r)
                    //     {
                    //         #pragma unroll
                    //         for (int c=0;c<3;++c) 
                    //         {
                    //             const double value = alpha[c][r] * qww;   // matches CPU: value[r][c] = alpha[c][r]*qww
                    //             const uint32_t MM_index = 4u + (uint32_t)(r*3 + c);

                    //             atomicAdd(&moments[MM_index*oneDensity + nodeIdx], (cudaMomentType)value);
                    //         }
                    //     }
                    // }

                    //? (i2>=0 && i2<2 && j2>=0 && j2<2 && k2>=0 && k2<2)
                    if ( (unsigned)i2 < 2u && (unsigned)j2 < 2u && (unsigned)k2 < 2u )
                    {
                        const int idx2 = i2*4 + j2*2 + k2; // 0..7 for (i2,j2,k2)

                        const double qww = q * q_dt_2mc * weights[idx1] * weights[idx2];

                        const uint32_t nodeIdx = toOneDimIndex((uint32_t)nxn,(uint32_t)nyn,(uint32_t)nzn, (uint32_t)ni,(uint32_t)nj,(uint32_t)nk);

                        //? Deposit 3x3 mass block into moment slots 4..12 (row-major: r*3+c)
                        atomicAdd(&moments[(4u + 0u)*oneDensity + nodeIdx], (cudaMomentType)(a00 * qww));
                        atomicAdd(&moments[(4u + 1u)*oneDensity + nodeIdx], (cudaMomentType)(a01 * qww));
                        atomicAdd(&moments[(4u + 2u)*oneDensity + nodeIdx], (cudaMomentType)(a02 * qww));
                        atomicAdd(&moments[(4u + 3u)*oneDensity + nodeIdx], (cudaMomentType)(a10 * qww));
                        atomicAdd(&moments[(4u + 4u)*oneDensity + nodeIdx], (cudaMomentType)(a11 * qww));
                        atomicAdd(&moments[(4u + 5u)*oneDensity + nodeIdx], (cudaMomentType)(a12 * qww));
                        atomicAdd(&moments[(4u + 6u)*oneDensity + nodeIdx], (cudaMomentType)(a20 * qww));
                        atomicAdd(&moments[(4u + 7u)*oneDensity + nodeIdx], (cudaMomentType)(a21 * qww));
                        atomicAdd(&moments[(4u + 8u)*oneDensity + nodeIdx], (cudaMomentType)(a22 * qww));
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
    const commonType xstart = grid->xStart, ystart = grid->yStart, zstart = grid->zStart;
    const uint32_t oneDensity = (uint32_t)nxn * (uint32_t)nyn * (uint32_t)nzn;

    // particle state
    const SpeciesParticle& pcl = pclsArray->getpcls()[pidx];
    const commonType x = pcl.get_x(), y = pcl.get_y(), z = pcl.get_z();
    const commonType u = pcl.get_u(), v = pcl.get_v(), w = pcl.get_w();
    const commonType q = pcl.get_q();

    commonType weights[8];
    int cx, cy, cz;
    grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);

    // previous cell index used by your packed field buffer
    const int prevIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;
    // we only need Bx,By,Bz (3 values) from each of 8 nodes
    commonType Bxl = 0, Byl = 0, Bzl = 0, Exl = 0, Eyl = 0, Ezl = 0;

    // fieldForPcls layout: for each node c in {0..7}, the 6-tuple is {Bx,By,Bz,Ex,Ey,Ez}
    // and the 8 nodes per cell make 24 entries per cell (8*3 for B + 8*3 for E).
    // index = prevIndex*24 + c*6 + i
    for (int icx = 0; icx < 8; ++icx) 
    {
        const int base = prevIndex * 24 + icx * 6;
        const commonType wc = weights[icx];
        Bxl += wc * fieldForPcls[base + 0];
        Byl += wc * fieldForPcls[base + 1];
        Bzl += wc * fieldForPcls[base + 2];
    }

    // ---------------- Rotation matrix alpha (relativistic-aware) ----------------
    //     double lorentz_factor = 1.0;

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
    const commonType Omx = q_dt_2mc * Bxl;
    const commonType Omy = q_dt_2mc * Byl;
    const commonType Omz = q_dt_2mc * Bzl;
    const commonType omsq = Omx*Omx + Omy*Omy + Omz*Omz;
    const commonType denom = (commonType)1.0 / ((commonType)1.0 + omsq);

    // alpha (same algebra as CPU)
    commonType a00 = ((commonType)1.0 + Omx*Omx) * denom;
    commonType a01 = (Omz + Omx*Omy)        * denom;
    commonType a02 = (-Omy + Omx*Omz)       * denom;

    commonType a10 = (-Omz + Omx*Omy)       * denom;
    commonType a11 = ((commonType)1.0 + Omy*Omy) * denom;
    commonType a12 = (Omx + Omy*Omz)        * denom;

    commonType a20 = (Omy + Omx*Omz)        * denom;
    commonType a21 = (-Omx + Omy*Omz)       * denom;
    commonType a22 = ((commonType)1.0 + Omz*Omz) * denom;

    // q * alpha * [u; v; w]  (F = 0 ignored)
    const commonType qau = q * (a00*u + a01*v + a02*w);
    const commonType qav = q * (a10*u + a11*v + a12*w);
    const commonType qaw = q * (a20*u + a21*v + a22*w);

    // --- node indices for CIC deposition (same mapping you use elsewhere) ---
    const int ix = 2 + int(floor((x - xstart) * grid->invdx));
    const int iy = 2 + int(floor((y - ystart) * grid->invdy));
    const int iz = 2 + int(floor((z - zstart) * grid->invdz));

    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz  );
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz  );
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz  );
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz  );
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);

    // --- deposit rho and implicit currents (first 4 moment channels) ---
    for (int cc = 0; cc < 8; ++cc) 
    {
        const commonType wc = weights[cc];
        const uint32_t idx = posIndex[cc];

        // Rho
        atomicAdd(&moments[0*oneDensity + idx], q * wc);
        
        // Jx, Jy, Jz
        atomicAdd(&moments[1*oneDensity + idx], qau * wc);
        atomicAdd(&moments[2*oneDensity + idx], qav * wc);
        atomicAdd(&moments[3*oneDensity + idx], qaw * wc);
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
    const commonType xstart = grid->xStart, ystart = grid->yStart, zstart = grid->zStart;
    const uint32_t oneDensity = (uint32_t)nxn * (uint32_t)nyn * (uint32_t)nzn;

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
        const int prevIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;
        // we only need Bx,By,Bz (3 values) from each of 8 nodes
        commonType Bxl = 0, Byl = 0, Bzl = 0, Exl = 0, Eyl = 0, Ezl = 0;

        // fieldForPcls layout: for each node c in {0..7}, the 6-tuple is {Bx,By,Bz,Ex,Ey,Ez}
        // and the 8 nodes per cell make 24 entries per cell (8*3 for B + 8*3 for E).
        // index = prevIndex*24 + c*6 + i
        //TODO: check this - PJD
        for (int icx = 0; icx < 8; ++icx) 
        {
            const int base = prevIndex * 24 + icx * 6;
            const commonType wc = weights[icx];
            Bxl += wc * fieldForPcls[base + 0];
            Byl += wc * fieldForPcls[base + 1];
            Bzl += wc * fieldForPcls[base + 2];
        }

        const commonType q_dt_2mc = (commonType)0.5 * momentParam->dt * momentParam->qom / momentParam->c;
        const commonType Omx = q_dt_2mc * Bxl;
        const commonType Omy = q_dt_2mc * Byl;
        const commonType Omz = q_dt_2mc * Bzl;
        const commonType omsq = Omx*Omx + Omy*Omy + Omz*Omz;
        const commonType denom = (commonType)1.0 / ((commonType)1.0 + omsq);

        // alpha (same algebra as CPU)
        commonType a00 = ((commonType)1.0 + Omx*Omx) * denom;
        commonType a01 = (Omz + Omx*Omy)        * denom;
        commonType a02 = (-Omy + Omx*Omz)       * denom;

        commonType a10 = (-Omz + Omx*Omy)       * denom;
        commonType a11 = ((commonType)1.0 + Omy*Omy) * denom;
        commonType a12 = (Omx + Omy*Omz)        * denom;

        commonType a20 = (Omy + Omx*Omz)        * denom;
        commonType a21 = (-Omx + Omy*Omz)       * denom;
        commonType a22 = ((commonType)1.0 + Omz*Omz) * denom;

        // q * alpha * [u; v; w]  (F = 0 ignored)
        const commonType qau = q * (a00*u + a01*v + a02*w);
        const commonType qav = q * (a10*u + a11*v + a12*w);
        const commonType qaw = q * (a20*u + a21*v + a22*w);

        //* ------------------------------------------------------------------- *//

        //? Map the 3D node index (i,j,k) into a flat index - node indices for CIC deposition
        const int ix = 2 + int(floor((x - xstart) * grid->invdx));
        const int iy = 2 + int(floor((y - ystart) * grid->invdy));
        const int iz = 2 + int(floor((z - zstart) * grid->invdz));

        uint32_t posIndex[8];
        posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz  );
        posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz-1);
        posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz  );
        posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz-1);
        posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz  );
        posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz-1);
        posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz  );
        posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);

        //* Deposit rho and implicit current (first 4 moments)
        for (int cc = 0; cc < 8; ++cc) 
        {
            const commonType wc = weights[cc];
            const uint32_t idx = posIndex[cc];

            //? Rho
            atomicAdd(&moments[0*oneDensity + idx], q * wc);
            
            //? Jxh, Jyh, Jzh
            atomicAdd(&moments[1*oneDensity + idx], qau * wc);
            atomicAdd(&moments[2*oneDensity + idx], qav * wc);
            atomicAdd(&moments[3*oneDensity + idx], qaw * wc);
        }

        //? Compute exact Mass Matrix
        exact_mass_matrix(moments, ix, iy, iz, q, q_dt_2mc, weights, a00, a01, a02, a10, a11, a12, a20, a21, a22, nxn, nyn, nzn);
    }
}