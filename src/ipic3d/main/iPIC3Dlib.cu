/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "mpi.h"
#include "MPIdata.h"
#include "iPic3D.h"
#include "TimeTasks.h"
#include "ipicdefs.h"
#include "debug.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
#include "Timing.h"
#include "ParallelIO.h"
#include "outputPrepare.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "Moments.h" // for debugging
#include "ECSIM_Moments.h"

#include "cudaTypeDef.cuh"
#include "momentKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "moverKernel.cuh"
#include "particleExchange.cuh"
#include "dataAnalysis.cuh"
#include "thread"
#include "future"
#include "particleControlKernel.cuh"

#include "../utility/LeXInt_Timer.hpp"

#ifdef USE_CATALYST
    #include "Adaptor.h"
#endif

using namespace iPic3D;

//! Destructor
c_Solver::~c_Solver()
{
    delete col;         // configuration parameters ("collectiveIO")
    delete vct;         // process topology
    delete grid;        // grid
    delete EMf;         // field

    #ifdef USE_ADIOS2
        delete adiosManager;
    #endif

    //* Delete particles
    if(particles)
    {
        for (int i = 0; i < ns; i++)
            particles[i].~Particles3D();

        free(particles);
    }

    if(outputPart) // initial and output particles
    {
        for (int i = 0; i < ns; i++)
            outputPart[i].~Particles3D();
        
        free(outputPart);
    }

    #ifdef USE_CATALYST
        Adaptor::Finalize();
    #endif

    delete [] kinetic_energy_species;
    delete [] bulk_energy_species;
    delete [] momentum_species;
    delete [] charge_species;
    delete [] num_particles_species;
    delete [] Qremoved;
    delete my_clock;
}

int c_Solver::Init(int argc, char **argv) 
{
    #if defined(__MIC__)
        assert_eq(DVECWIDTH,8);
    #endif

    Parameters::init_parameters();
    nprocs = MPIdata::get_nprocs();
    myrank = MPIdata::get_rank();

    col = new Collective(argc, argv);
    restart_cycle = col->getRestartOutputCycle();
    SaveDirName = col->getSaveDirName();
    RestartDirName = col->getRestartDirName();
    restart_status = col->getRestart_status();
    ns = col->getNs();                          // number of particle species involved in simulation
    first_cycle = col->getLast_cycle() + 1;     // get the last cycle from the restart

    vct = new VCtopology3D(*col);
  
    //? Check if we can map the processes into a matrix ordering defined in Collective.cpp
    if (nprocs != vct->getNprocs())
        if (myrank == 0) 
        {
            cerr << "Error: " << nprocs << " processes cant be mapped as " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << ". Change XLEN, YLEN, & ZLEN in input file. " << endl;
            MPIdata::instance().finalize_mpi();
            return (1);
        }

    //* Create a new communicator with a 3D virtual Cartesian topology
    vct->setup_vctopology(MPIdata::get_PicGlobalComm());

    #ifdef BATSRUS
        // set index offset for each processor
        col->setGlobalStartIndex(vct);
    #endif

    //* Print initial settings to stdout and a file
    if (myrank == 0) 
    {
        // check and create the output directory, only if it is not a restart run
        if(restart_status == 0)
        {
            checkOutputFolder(SaveDirName); 
            
            if(RestartDirName != SaveDirName)
                checkOutputFolder(RestartDirName); 
        }
        
        MPIdata::instance().Print();
        vct->Print();
        col->Print();
        col->save();
    }

    //* Create local grid
    grid = new Grid3DCU(col, vct);              // Create the local grid
    EMf = new EMfields3D(col, grid, vct);       // Create Electromagnetic Fields Object

    //! =============================== INITIAL FIELD DISTRIBUTION =============================== !//

    if (restart_status == 1 || restart_status == 2)
    {   
        //! RESTART
        EMf->init();
    }
    else if (restart_status == 0)
    {   
        //! NEW INITIAL CONDITION (FIELDS)
        if (col->getRelativistic())
        {
            //! Relativistic Cases
            if      (col->getCase()=="Relativistic_Double_Harris_pairs")            EMf->init_Relativistic_Double_Harris_pairs();
            else if (col->getCase()=="Relativistic_Double_Harris_ion_electron")     EMf->init_Relativistic_Double_Harris_ion_electron();
            else if (col->getCase()=="Shock1D")                                     EMf->initShock1D();
            else if (col->getCase()=="Double_Harris")                               EMf->init_double_Harris();              //* Works for small enough velocities
            else if (col->getCase()=="Maxwell_Juttner")                             EMf->init();
            else 
            {
                if (myrank==0)
                {
                    cout << " =================================================================== " << endl;
                    cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
                    cout << "     Runing relativistic simulation with the default initialisation. " << endl;
                    cout << " =================================================================== " << endl;
                }

                EMf->init();
            }
        }
        else
        {
            //! Non Relativistic Cases
            if      (col->getCase()=="GEMnoPert") 		            EMf->initGEMnoPert();
            else if (col->getCase()=="ForceFree") 		            EMf->initForceFree();
            else if (col->getCase()=="GEM")       		            EMf->initGEM();
            else if (col->getCase()=="Double_Harris")               EMf->init_double_Harris();
            else if (col->getCase()=="Double_Harris_Hump")          EMf->init_double_Harris_hump();
            #ifdef BATSRUS
                else if (col->getCase()=="BATSRUS")   		        EMf->initBATSRUS();
            #endif
            else if (col->getCase()=="Dipole")    		            EMf->initDipole();
            else if (col->getCase()=="Dipole2D")  		            EMf->initDipole2D();
            else if (col->getCase()=="NullPoints")             	    EMf->initNullPoints();
            else if (col->getCase()=="TaylorGreen")                 EMf->initTaylorGreen();
            else if (col->getCase()=="RandomCase")                  EMf->initRandomField();
            else if (col->getCase()=="Uniform")                     EMf->init();
            else if (col->getCase()=="Maxwellian")                  EMf->init();
            else if (col->getCase()=="KHI_FLR")                     EMf->init_KHI_FLR();
            else 
            {
                if (myrank==0) 
                {
                    cout << " ================================================================ " << endl;
                    cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
                    cout << "       Runing simulation with the default initialisation. " << endl;
                    cout << " ================================================================ " << endl;
                }
                
                EMf->init();
            }
        }
    }
    else
    {
        if (myrank==0)
        {
            cout << "Incorrect restart status!" << endl;
            cout << "restart_status = 0 ---> NO RESTART!" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are different" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are the same" << endl;
        }
        abort();
    }

    //! =============== INITIAL PARTICLE DISTRIBUTION (if NOT starting from RESTART) =============== !//

    //* Allocation of particles
    particles = (Particles3D*) malloc(sizeof(Particles3D)*ns);
   
    for (int is = 0; is < ns; is++)
    {
        new(&particles[is]) Particles3D(is, col, vct, grid);
        const auto totalPcl = col->getNpcel(is) * grid->getNXN() * grid->getNYN() * grid->getNZN();
        
        particles[is].reserveSpace(totalPcl * 0.1);     // reserve the size for exchange
        particles[is].get_pcl_array().clear();
    }

    outputPart = (Particles3D*) malloc(sizeof(Particles3D)*ns);
    for (int is = 0; is < ns; is++)
    {
        new(&outputPart[is]) Particles3D(is, col, vct, grid);
        const auto totalPcl = col->getNpcel(is) * grid->getNXN() * grid->getNYN() * grid->getNZN();

        if (restart_status == 0)
        {
            outputPart[is].reserveSpace(totalPcl); 
            outputPart[is].get_pcl_array().clear();
        } 
        else 
        {   
            // restart
            outputPart[is].restartLoad();
        }
    }

    #ifdef USE_ADIOS2
        adiosManager = new ADIOS2IO::ADIOS2Manager();
        adiosManager->initOutputFiles(""s, col->getParticlesOutputCycle()? col->getPclOutputTag() : ""s, 0, *this);
    #endif

    //! NEW INITIAL CONDITION (PARTICLES)
    if (restart_status == 0)
    {
        for (int i = 0; i < ns; i++)
        {
            if (col->getRelativistic())
			{
                //! Relativistic Cases
				// if      (col->getCase()=="Relativistic_Double_Harris_pairs") 	        outputPart[i].Relativistic_Double_Harris_pairs(EMf);
				// else if (col->getCase()=="Relativistic_Double_Harris_ion_electron") 	outputPart[i].Relativistic_Double_Harris_ion_electron(EMf);
				// else if (col->getCase()=="Shock1D") 	                                outputPart[i].Shock1D(EMf);
				// else if (col->getCase()=="Shock1D_DoublePiston") 	                    outputPart[i].Shock1D_DoublePiston(EMf);
                // else if (col->getCase()=="Maxwell_Jutter") 	                            outputPart[i].Maxwell_Juttner(EMf);
                // else if (col->getCase()=="Double_Harris")                               outputPart[i].maxwellian_Double_Harris(EMf);           //* Works for small enough velocities
				// else                                                                    outputPart[i].Maxwell_Juttner(EMf);
			}
            else
            {
                //! Non Relativistic Cases
                if      (col->getCase()=="ForceFree") 		                            outputPart[i].force_free(EMf);
                #ifdef BATSRUS
                    else if (col->getCase()=="BATSRUS")   		                        outputPart[i].MaxwellianFromFluid(EMf, col, i);
                #endif
                else if (col->getCase()=="NullPoints")    	                            outputPart[i].maxwellianNullPoints(EMf);
                else if (col->getCase()=="TaylorGreen")                                 outputPart[i].maxwellianNullPoints(EMf);
                else if (col->getCase()=="Uniform")    	                                outputPart[i].uniform_background(EMf);
                else if (col->getCase()=="Double_Harris")                               outputPart[i].maxwellian_Double_Harris(EMf);
                else if (col->getCase()=="Double_Harris_Hump")                          outputPart[i].maxwellian_Double_Harris(EMf);
                else if (col->getCase()=="Maxwellian") 		                            outputPart[i].maxwellian(EMf);
                else if (col->getCase()=="KHI_FLR")                                     outputPart[i].maxwellian_KHI_FLR(EMf);
                else                                  		                            outputPart[i].maxwellian(EMf);
            }

            outputPart[i].reserve_remaining_particle_IDs();
            outputPart[i].fixPosition();
        }
    }

    //* Allocate test particles (if any)
    nstestpart = col->getNsTestPart();
    if(nstestpart>0)
    {
        testpart = (Particles3D*) malloc(sizeof(Particles3D)*nstestpart);
        for (int i = 0; i < nstestpart; i++)
        {
            new(&testpart[i]) Particles3D(i+ns,col,vct,grid);//species id for test particles is increased by ns
            testpart[i].pitch_angle_energy(EMf);
        }
    }

    //TODO PJD: check how data is written to ADIOS2
    //? Write particle and field output data
    if (Parameters::get_doWriteOutput())
    {
        if(!col->field_output_is_off())
        {
            if(col->getWriteMethod()=="pvtk")
            {
                if(!(col->getFieldOutputTag()).empty())
                    fieldwritebuffer = newArr4(float,(grid->getNZN()-3),grid->getNYN()-3,grid->getNXN()-3,3);
                
                if(!(col->getMomentsOutputTag()).empty())
                    momentwritebuffer = newArr3(float,(grid->getNZN()-3), grid->getNYN()-3, grid->getNXN()-3);
            }
            else if(col->getWriteMethod()=="nbcvtk")
            {
                momentreqcounter = 0;
                fieldreqcounter  = 0;

                if(!(col->getFieldOutputTag()).empty())
                    fieldwritebuffer = newArr4(float,(grid->getNZN()-3)*4,grid->getNYN()-3,grid->getNXN()-3,3);

                if(!(col->getMomentsOutputTag()).empty())
                    momentwritebuffer = newArr3(float,(grid->getNZN()-3)*14, grid->getNYN()-3, grid->getNXN()-3);
            }
        }
    }

    //? Write conserved parameters to files
    kinetic_energy_species = new double[ns];
    bulk_energy_species = new double[ns];
    num_particles_species = new int[ns];
    momentum_species = new double[ns];
    charge_species = new double[ns];
    
    cq = SaveDirName + "/ConservedQuantities.txt";
    cqs = SaveDirName + "/SpeciesQuantities.txt";
    if (myrank == 0 && restart_status == 0) 
    {
        ofstream my_file(cq.c_str());
        my_file.close();

        ofstream my_file_(cqs.c_str());
        my_file_.close();
    }

    Qremoved = new double[ns];

    #ifdef USE_CATALYST
    Adaptor::Initialize(col, \
                        (int)(grid->getXstart()/grid->getDX()), \
                        (int)(grid->getYstart()/grid->getDY()), \
                        (int)(grid->getZstart()/grid->getDZ()), \
                        grid->getNXN(),
                        grid->getNYN(),
                        grid->getNZN(),
                        grid->getDX(),
                        grid->getDY(),
                        grid->getDZ());
    #endif

    //* Create or open the particle number file csv
    pclNumCSV = std::ofstream(SaveDirName + "/particleNum" + std::to_string(myrank) + ".csv", std::ios::app); 
    pclNumCSV << "cycle,";
    for(int i=0; i<ns-1; i++)
        pclNumCSV << "species" << i << ",";
    pclNumCSV << "species" << ns-1 << std::endl;

    //! Initialise CUDA
    initCUDA();

    my_clock = new Timing(myrank);

    return 0;
}

//! ============================================= CUDA ============================================= !//

int c_Solver::initCUDA()
{
    LeXInt::timer time_init_cuda;

    time_init_cuda.start();

    //? Compute per-node mapping from MPI processes to GPUs
    {
        MPI_Comm sharedComm; int sharedRank, sharedSize; int deviceOnNode;          //* locals: per-node communicator, local rank & size, and GPU count
        MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), 
                            MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm);   //* split the global comm by shared-memory domain (i.e., per node)
        MPI_Comm_rank(sharedComm, &sharedRank);                                     //* get this process’s rank within the node
        MPI_Comm_size(sharedComm, &sharedSize);                                     //* Number of MPI processes on this node
        cudaErrChk(cudaGetDeviceCount(&deviceOnNode));                              //* how many CUDA devices are visible on this node
        
        if(sharedSize <= deviceOnNode)
            cudaDeviceOnNode = sharedRank;                                          //* 1:1 mapping when MPI processes <= GPUs; rank "i" uses GPU "i"                          
        else
        {
            if(sharedSize % deviceOnNode != 0)
            {   
                //* If processes per node is not divisible by GPUs
                cout << "Error: Can not map process to device on node. " << endl;
                cout << "Global COMM rank: " << MPIdata::get_rank() << " Shared COMM size: " << sharedSize << " Device Number in Node: " << deviceOnNode << endl;
               
                MPIdata::instance().finalize_mpi();
                return (1);
            }
            
            //? Each GPU gets "procPerDevice" MPI processes
            int procPerDevice = sharedSize / deviceOnNode;                          //* Compute how many MPI processes will share each GPU
            cudaDeviceOnNode = sharedRank / procPerDevice;                          //* Map local rank to GPU index (block assignment - contiguous chunks of ranks)
        }
        
        cudaErrChk(cudaSetDevice(cudaDeviceOnNode));                                //* Select the chosen GPU for this MPI process

        #ifndef NDEBUG
            if(sharedRank == 0)
                cout << "[*]GPU assignment: shared comm size: " << sharedSize << " GPU device on the node: " << deviceOnNode << endl;
        #endif
    }

    //* ===================== Initialise Particles on Device ===================== *//
  
	//* Initialise 2 CUDA streams per species (overlap work to improve concurrency)
    streams = new cudaStream_t[ns*2]; stayedParticle = new int[ns]; exitingResults = new std::future<int>[ns];
    for(int is=0; is<ns; is++)
    { 
        cudaErrChk(cudaStreamCreate(streams+is));        // Stream A for species i
        cudaErrChk(cudaStreamCreate(streams+is+ns));     // Stream B for species i
        stayedParticle[is] = 0;                          // Stores how many particles remained in the local MPI subdomain after the particle pusher has been called
    }
	
    //? Initialise contiguous arrays of "ns" pointers on device
    //* Pointer to the particle arrays
    pclsArrayHostPtr = new particleArrayCUDA*[ns];          
    pclsArrayCUDAPtr = new particleArrayCUDA*[ns];
    
    //* Arrays that mark particle exchange across MPI subdomains
    departureArrayHostPtr = new departureArrayType*[ns];
    departureArrayCUDAPtr = new departureArrayType*[ns];

    //* Small counters/offsets on the device used to group particles by destination (e.g., -X, +X, -Y, …)
    hashedSumArrayHostPtr = new hashedSum*[ns];
    hashedSumArrayCUDAPtr = new hashedSum*[ns];
    
    //* A compact list of indices of exiting particles
    exitingArrayHostPtr = new exitingArray*[ns];
    exitingArrayCUDAPtr = new exitingArray*[ns];

    //* Temporary staging memory for the packed particles before appending to main arrays
    fillerBufferArrayHostPtr = new fillerBuffer*[ns];
    fillerBufferArrayCUDAPtr = new fillerBuffer*[ns];

    //* Loop over species
    for(int i=0; i<ns; i++)
    {
        // the constructor will copy particles from host to device
        pclsArrayHostPtr[i] = newHostPinnedObject<particleArrayCUDA>(outputPart+i, 1.4, streams[i]);    //* Allocate particle array in host-pinned memory; use the outputPart as the initial pcls
        pclsArrayHostPtr[i]->setInitialNOP(pclsArrayHostPtr[i]->getNOP());                              //* Remember initial number of particles (NOP)
        pclsArrayCUDAPtr[i] = pclsArrayHostPtr[i]->copyToDevice();                                      //* Create device mirror of particle array (async if stream used)

        //* Allocate array in host-pinned memory and copy this to Device
        departureArrayHostPtr[i] = newHostPinnedObject<departureArrayType>(pclsArrayHostPtr[i]->getSize()); // same length
        departureArrayCUDAPtr[i] = departureArrayHostPtr[i]->copyToDevice();
        
        //* Asynchronously zero-initialize the departures buffer on CUDA stream i (pointer must be device/unified or mapped)
        cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType), streams[i]));

        //TODO: ChatGPT recommended the following line instead of the above -- check
        // cudaErrChk(cudaMemsetAsync(departureArrayCUDAPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize()*sizeof(departureArrayElementType), streams[i]));

        // hashedSumArrayHostPtr[i] = new hashedSum[8]{ // 
        //   hashedSum(5), hashedSum(5), hashedSum(5), hashedSum(5), 
        //   hashedSum(5), hashedSum(5), hashedSum(10), hashedSum(10)
        // };

        //* Allocate array in host-pinned memory and copy this to Device 
        hashedSumArrayHostPtr[i] = newHostPinnedObjectArray<hashedSum>(departureArrayElementType::HASHED_SUM_NUM, 10);
        hashedSumArrayCUDAPtr[i] = copyArrayToDevice(hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);
        
        //* Allocate array in host-pinned memory and copy this to Device 
        exitingArrayHostPtr[i] = newHostPinnedObject<exitingArray>(0.1 * pclsArrayHostPtr[i]->getNOP());        //* 0.1 -> Assuming only few particles would move across MPI domains
        exitingArrayCUDAPtr[i] = exitingArrayHostPtr[i]->copyToDevice();
        
        //* Allocate array in host-pinned memory and copy this to Device 
        fillerBufferArrayHostPtr[i] = newHostPinnedObject<fillerBuffer>(0.1 * pclsArrayHostPtr[i]->getNOP());   //* 0.1 -> Assuming only few particles would move across MPI domains
        fillerBufferArrayCUDAPtr[i] = fillerBufferArrayHostPtr[i]->copyToDevice();
    }

    //* Allocate array in host-pinned memory and copy this to Device; only one grid for all species
    grid3DCUDAHostPtr = newHostPinnedObject<grid3DCUDA>(grid);
    grid3DCUDACUDAPtr = copyToDevice(grid3DCUDAHostPtr, 0);

    moverParamHostPtr = new moverParameter*[ns];        // allocate host array of pointers for each species
    moverParamCUDAPtr = new moverParameter*[ns];        // allocate host array of device pointers for each species

    //* Loop over species
    for (int i=0; i<ns; i++)
    {
        //* Allocate host-pinned Particle Mover params for each species
        moverParamHostPtr[i] = newHostPinnedObject<moverParameter>(outputPart+i, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i], hashedSumArrayCUDAPtr[i]);

        //* Initialise moverParam for OpenBC, repopulateInjection, sphere
        outputPart[i].openbc_particles_outflowInfo(&moverParamHostPtr[i]->doOpenBC, moverParamHostPtr[i]->applyOpenBC, moverParamHostPtr[i]->deleteBoundary, moverParamHostPtr[i]->openBoundary);
        
        moverParamHostPtr[i]->appendCountAtomic = 0;

        //* Particle Injection
        if(col->getRHOinject(i)>0.0)
            outputPart[i].repopulate_particlesInfo(&moverParamHostPtr[i]->doRepopulateInjection, moverParamHostPtr[i]->doRepopulateInjectionSide, moverParamHostPtr[i]->repopulateBoundary);
        else 
            moverParamHostPtr[i]->doRepopulateInjection = false;        // disable repopulation

        // if (col->getCase() == "Dipole") 
        // {
        //     moverParamHostPtr[i]->doSphere = 1;
        //     moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center();
        //     moverParamHostPtr[i]->sphereOrigin[1] = col->gety_center();
        //     moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center();
        //     moverParamHostPtr[i]->sphereRadius = col->getL_square();
        // }
        // else if (col->getCase() == "Dipole2D") 
        // {
        //     moverParamHostPtr[i]->doSphere = 2;
        //     moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center();
        //     moverParamHostPtr[i]->sphereOrigin[1] = 0.0;
        //     moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center();
        //     moverParamHostPtr[i]->sphereRadius = col->getL_square();
        // } 
        // else
        //     moverParamHostPtr[i]->doSphere = 0;

        //? Copy Particle Mover params to device on stream i
        moverParamCUDAPtr[i] = copyToDevice(moverParamHostPtr[i], streams[i]);
    }

    //* ===================== Initialise Moments on Device ===================== *//

    momentParamHostPtr = new momentParameter*[ns];      // allocate host array of pointers for each species
    momentParamCUDAPtr = new momentParameter*[ns];      // allocate host array of device pointers for each species

    for(int i=0; i<ns; i++)
    {
        //* Allocate host-pinned Moment Gatherer params for each species
        momentParamHostPtr[i] = newHostPinnedObject<momentParameter>(pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);

        //* Params needed for ECSIM/RelSIM moments computation
        momentParamHostPtr[i]->dt  = col->getDt();
        momentParamHostPtr[i]->qom = col->getQOM(i);
        momentParamHostPtr[i]->c   = col->getC();

        //* Optional: Only for RelSIM
        // momentParamHostPtr[i]->isRelativistic = opts.relativistic;
        // momentParamHostPtr[i]->relPusher      = opts.relPusher;         //* 0: Boris, 1: Lapenta-Markidis,

        //? Copy Moment Gatherer params to device on stream i
        momentParamCUDAPtr[i] = copyToDevice(momentParamHostPtr[i], streams[i]);
    }

    auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();       //* Total number of grid cells
    momentsCUDAPtr = new cudaMomentType*[ns];                               //* Host array holding per-species device pointers for moments
    constexpr int ECSIM_CHANNELS = 4 + 14*9;                                //* rho, Jxh, Jyh, Jzh, 9 * 14 mass matrix components (27 in total - symmetric, so 14)
    
    for(int i=0; i<ns; i++)
    {
        cudaErrChk(cudaMalloc(&(momentsCUDAPtr[i]), gridSize*ECSIM_CHANNELS*sizeof(cudaMomentType)));                       //* Assign moments arrays
        cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0, gridSize*ECSIM_CHANNELS*sizeof(cudaMomentType), streams[i]));      //* Set all moments arrays to 0
    }

    // Device outputs for the 9 distinct mass components (per species)
    dMxx = new cudaCommonType*[ns]; dMxy = new cudaCommonType*[ns]; dMxz = new cudaCommonType*[ns];
    dMyx = new cudaCommonType*[ns]; dMyy = new cudaCommonType*[ns]; dMyz = new cudaCommonType*[ns];
    dMzx = new cudaCommonType*[ns]; dMzy = new cudaCommonType*[ns]; dMzz = new cudaCommonType*[ns];

    for (int i = 0; i < ns; ++i) 
    {
        cudaErrChk(cudaMalloc(&dMxx[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMxy[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMxz[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMyx[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMyy[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMyz[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMzx[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMzy[i], gridSize * sizeof(cudaCommonType)));
        cudaErrChk(cudaMalloc(&dMzz[i], gridSize * sizeof(cudaCommonType)));

        // (optional) zero them; not strictly required if every node gets written each step
        // cudaErrChk(cudaMemsetAsync(dMxx[i], 0, gridSize*sizeof(cudaCommonType), streams[i]));
        // ... repeat memset for the others if you want a clean start
    }

    for(int i=0; i<ns; i++)
    {
        //* ρ and implicit current J_hat
        cudaErrChk(cudaHostRegister((void*)&(EMf->getRHOns().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getJxhs().get(i,0,0,0)),  gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getJyhs().get(i,0,0,0)),  gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getJzhs().get(i,0,0,0)),  gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));

        //* Nine mass-matrix components
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMxx().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMxy().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMxz().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMyx().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMyy().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMyz().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMzx().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMzy().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
        cudaErrChk(cudaHostRegister((void*)&(EMf->getMzz().get(i,0,0,0)),   gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
    }

    //* ===================== Initialise Field Grid on Device ===================== *//

    const int fieldSize = grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1);
    cudaMalloc(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType));
    cudaErrChk(cudaHostAlloc((void**)&fieldForPclHostPtr, fieldSize * 24 * sizeof(cudaFieldType), 0));

    threadPoolPtr = new ThreadPool(ns);
    cudaErrChk(cudaEventCreateWithFlags(&event0, cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&eventOutputCopy, cudaEventDisableTiming|cudaEventBlockingSync));

    //* Particle Merging
    toBeMerged = new int[2 * ns];
    for(int i=0;i<2*ns;i++)
        toBeMerged[i] = 0;

    cudaErrChk(cudaHostAlloc(&cellCountHostPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC(), cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&cellOffsetHostPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC(), cudaHostAllocDefault));

    cudaErrChk(cudaMalloc(&cellCountCUDAPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC()));
    cudaErrChk(cudaMalloc(&cellOffsetCUDAPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC()));

    dataAnalysis::dataAnalysisPipeline::createOutputDirectory(myrank, ns, vct);

    cudaErrChk(cudaDeviceSynchronize());

    if (MPIdata::get_rank() == 0)
        cout << endl << "Initialisation on GPU completed in "  << time_init_cuda.total()   << " s" << endl;

    return 0;
}

//* Delete all CUDA-related arrays, free memory
int c_Solver::deInitCUDA()
{
    cudaEventDestroy(event0);
    cudaEventDestroy(eventOutputCopy);

    delete threadPoolPtr;

    deleteHostPinnedObject(grid3DCUDAHostPtr);
    cudaFree(grid3DCUDACUDAPtr);

    cudaFree(fieldForPclCUDAPtr);
    cudaFreeHost(fieldForPclHostPtr);

    //* Release device objects
    for(int i=0; i<ns; i++)
    {
        //* ==================  Delete host object, deconstruct ==================

        deleteHostPinnedObject(pclsArrayHostPtr[i]);
        deleteHostPinnedObject(departureArrayHostPtr[i]);
        deleteHostPinnedObjectArray(hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);
        deleteHostPinnedObject(exitingArrayHostPtr[i]);
        deleteHostPinnedObject(fillerBufferArrayHostPtr[i]);
        deleteHostPinnedObject(moverParamHostPtr[i]);
        deleteHostPinnedObject(momentParamHostPtr[i]);

        //* ==================  cudaFree device object memory ==================

        cudaFree(pclsArrayCUDAPtr[i]);
        cudaFree(departureArrayCUDAPtr[i]);
        cudaFree(hashedSumArrayCUDAPtr[i]);
        cudaFree(exitingArrayCUDAPtr[i]);
        cudaFree(fillerBufferArrayCUDAPtr[i]);
        cudaFree(moverParamCUDAPtr[i]);
        cudaFree(momentParamCUDAPtr[i]);
        cudaFree(momentsCUDAPtr[i]);

        cudaFree(dMxx[i]); cudaFree(dMxy[i]); cudaFree(dMxz[i]);
        cudaFree(dMyx[i]); cudaFree(dMyy[i]); cudaFree(dMyz[i]);
        cudaFree(dMzx[i]); cudaFree(dMzy[i]); cudaFree(dMzz[i]);
    }

    //* Delete pointer arrays
    delete[] pclsArrayHostPtr;
    delete[] pclsArrayCUDAPtr;
    delete[] departureArrayHostPtr;
    delete[] departureArrayCUDAPtr;
    delete[] hashedSumArrayHostPtr;
    delete[] hashedSumArrayCUDAPtr;
    delete[] exitingArrayHostPtr;
    delete[] exitingArrayCUDAPtr;
    delete[] fillerBufferArrayHostPtr;
    delete[] fillerBufferArrayCUDAPtr;
    delete[] moverParamHostPtr;
    delete[] moverParamCUDAPtr;
    delete[] momentParamHostPtr;
    delete[] momentParamCUDAPtr;
    delete[] momentsCUDAPtr;
    delete[] toBeMerged;

    delete[] dMxx; delete[] dMxy; delete[] dMxz;
    delete[] dMyx; delete[] dMyy; delete[] dMyz;
    delete[] dMzx; delete[] dMzy; delete[] dMzz;

    //* Delete streams
    for(int i=0; i<ns*2; i++)
        cudaStreamDestroy(streams[i]);

    delete[] streams;
    delete[] stayedParticle;
    delete[] exitingResults;

    //* Deregister the pinned memory
    for (int i = 0; i < ns; i++) 
    {
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getRHOns().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getJxhs().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getJyhs().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getJzhs().get(i,0,0,0))));

        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMxx().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMxy().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMxz().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMyx().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMyy().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMyz().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMzx().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMzy().get(i,0,0,0))));
        cudaErrChk(cudaHostUnregister((void*)&(EMf->getMzz().get(i,0,0,0))));
    }

    return 0;
}


//! =============================================================== iPIC3D MAIN COMPUTATION =============================================================== !//

//! Compute moments
void c_Solver::CalculateMoments() 
{
    auto oneDensity = (uint64_t)grid->getNXN() * (uint64_t)grid->getNYN() * (uint64_t)grid->getNZN();

    //? Get the field data
    EMf->set_fieldForPclsToCenter(fieldForPclHostPtr);

    //? Asynchronously copy the field data from Host --> Device
    cudaEvent_t fieldReady;
    cudaErrChk(cudaEventCreateWithFlags(&fieldReady, cudaEventDisableTiming));
    
    cudaErrChk(cudaMemcpyAsync(fieldForPclCUDAPtr, fieldForPclHostPtr, (grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1)) * 24 * sizeof(cudaFieldType), cudaMemcpyDefault, streams[0]));

    // Record “field is ready” on streams[0]
    cudaErrChk(cudaEventRecord(fieldReady, streams[0]));

    for(int i=0; i<ns; i++)
    {
        //* Ensure field is ready
        cudaStreamWaitEvent(streams[i], fieldReady, 0);

        //* Set all moments to 0
        const size_t bytes = (size_t)moments130::NUM_CHANNELS * oneDensity * sizeof(cudaMomentType);
        cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0, bytes, streams[i]));
        
        //! Deposit moments from particles (for the very first time)
        ECSIM_RelSIM_Moments_PostExchange <<< (pclsArrayHostPtr[i]->getNOP()/256 + 1), 256, 0, streams[i] >>> (momentParamCUDAPtr[i], grid3DCUDACUDAPtr, fieldForPclCUDAPtr, momentsCUDAPtr[i], 0);

        //! Remove - PJD
        cudaErrChk(cudaDeviceSynchronize());

        cudaErrChk(cudaGetLastError());
        cudaErrChk(cudaDeviceSynchronize());

        // Dump a few entries from device CH_JX and CH_JZ directly
        {
            constexpr int N = 64; // first 64 nodes
            std::vector<cudaMomentType> hJx(N), hJz(N);

            const auto offJx = (size_t)moments130::chan_offset(moments130::CH_JX, oneDensity);
            const auto offJz = (size_t)moments130::chan_offset(moments130::CH_JZ, oneDensity);

            cudaErrChk(cudaMemcpy(hJx.data(),
                                momentsCUDAPtr[i] + offJx,
                                N * sizeof(cudaMomentType),
                                cudaMemcpyDeviceToHost));

            cudaErrChk(cudaMemcpy(hJz.data(),
                                momentsCUDAPtr[i] + offJz,
                                N * sizeof(cudaMomentType),
                                cudaMemcpyDeviceToHost));

            std::cout.setf(std::ios::scientific);
            std::cout << std::setprecision(16);

            std::cout << "[DEBUG] Device moments CH_JX first " << N << ":\n";
            for (int t = 0; t < N; ++t) std::cout << hJx[t] << " ";
            std::cout << "\n";

            std::cout << "[DEBUG] Device moments CH_JZ first " << N << ":\n";
            for (int t = 0; t < N; ++t) std::cout << hJz[t] << " ";
            std::cout << "\n";
        }

        //* Copy moments back to field object EMf (Device --> Host)
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_RHO, oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JX, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJyhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JY, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JZ, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));

        for (int ind = 0; ind < moments130::NUM_NEIGHBORS; ++ind)
        {
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        }

        //! Remove - PJD
        cudaErrChk(cudaDeviceSynchronize());

        if (vct->getCartesian_rank() == 0) 
        {
            std::cout << "Jxhs(species 0) k=0\n";
            for (int I=0; I<grid->getNXN(); ++I) {
                for (int J=0; J<grid->getNYN(); ++J) {
                    std::cout << EMf->getJxhs().get(0,I,J,0) << " ";
                }
                std::cout << "\n";
            }
        }
    }

    //* Synchronise
    Finalise_Moments();

    cudaErrChk(cudaEventDestroy(fieldReady));
}

//! Summations and communication of moment data needed by next field solve
void c_Solver::Finalise_Moments() 
{
    timeTasks_set_main_task(TimeTasks::MOMENTS);

    #ifdef __PROFILING__
    LeXInt::timer time_cm, time_com, time_int, time_total;

    time_total.start();
    #endif

    //? Synchronise
    cudaErrChk(cudaDeviceSynchronize());

    //! Particle Merging
    // constexpr bool PARTICLE_MERGING = true;
    // if constexpr(PARTICLE_MERGING)
    // {
    //     // check which one to merge
    //     for(int i = 0; i < ns; i++) {
    //     if(pclsArrayHostPtr[i]->getNOP() > 1.05 * pclsArrayHostPtr[i]->getInitialNOP()) {
    //         toBeMerged[2 * i] = 1;
    //     }
    //     else{
    //         toBeMerged[2 * i] = 0;
    //     }
    //     }
    //     // select spcecies to merge: the one that has not been merged for most cycles among the species that require merging
    //     mergeIdx = -1;
    //     int mergeCountFromLast = -1;
    //     for(int i=0;i<ns;i++){
    //         if( (toBeMerged[2 * i] == 1) && (toBeMerged[2 * i + 1] > mergeCountFromLast) ){
    //         mergeIdx = i;
    //         mergeCountFromLast = toBeMerged[2 * i + 1];
    //         }
    //     }

    //     if (mergeIdx >= 0 && mergeIdx < ns){
    //     const auto& i = mergeIdx; 

    //     if(outputPart[i].get_pcl_array().capacity() < pclsArrayHostPtr[i]->getNOP()){
    //         auto pclArray = outputPart[i].get_pcl_arrayPtr();
    //         pclArray->reserve(pclsArrayHostPtr[i]->getNOP() * 1.2);
    //     }
    //     cudaErrChk(cudaMemcpyAsync(outputPart[i].get_pcl_array().getList(), pclsArrayHostPtr[i]->getpcls(), 
    //                                 pclsArrayHostPtr[i]->getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[i]));
    //     outputPart[i].get_pcl_array().setSize(pclsArrayHostPtr[i]->getNOP()); 
    //     }
    // }
    // else
    // {
    //     mergeIdx = -1; 
    // }

    #ifdef __PROFILING__
    time_com.start();
    #endif

    //? Communicate moments
    for (int is = 0; is < ns; is++)
        EMf->communicateGhostP2G_ecsim(is);

    EMf->communicateGhostP2G_mass_matrix();

    #ifdef __PROFILING__
    time_com.stop();
    #endif

    #ifdef __PROFILING__
    time_int.start();
    #endif
    
    //? Sum over all the species (charge and current densities)
    EMf->sumOverSpecies();

    cudaErrChk(cudaDeviceSynchronize());

    //? Communicate average densities
    for (int is = 0; is < ns; is++)
        EMf->interpolateCenterSpecies(is);

    #ifdef __PROFILING__
    time_int.stop();
    #endif

    EMf->timeAveragedRho(col->getPoissonMArho());
    EMf->timeAveragedDivE(col->getPoissonMAdiv());

    #ifdef __PROFILING__
    time_total.stop();

    if (MPIdata::get_rank() == 0)
    {
        cout << endl << "Profiling of Finalise_Moments" << endl; 
        cout << "Communicate moments         : " << time_com.total()   << " s, fraction of time taken in Finalise_Moments(): " << time_com.total()/time_total.total() << endl;
        cout << "Summation & interpolation   : " << time_int.total()   << " s, fraction of time taken in Finalise_Moments(): " << time_int.total()/time_total.total() << endl;
        cout << "Finalise_Moments()          : " << time_total.total() << " s" << endl << endl;
    }
    #endif
}


//! Compute electromagnetic field
void c_Solver::Compute_EM_Fields(int cycle)
{
    col->setCurrentCycle(cycle);

    #ifdef __PROFILING__
    LeXInt::timer time_e, time_b, time_div, time_total;
    
    time_total.start();
    #endif

    //TODO: Only needed for the cases "Shock1D_DoublePiston" and "LangevinAntenna"; TBD later
	//* Update external fields to n+1/2
	// EMf->updateExternalFields(vct, grid, col, cycle); 

    //TODO: Only needed for the cases "ForcedDynamo"; TBD later
	//* Update particle external forces to n+1/2
	// EMf->updateParticleExternalForces(vct, grid, col, cycle); 

    #ifdef __PROFILING__
    time_e.start();
    #endif
    
    //? Compute E
    EMf->calculateE();
    
    #ifdef __PROFILING__
    time_e.stop();
    #endif
    
    #ifdef __PROFILING__
    time_b.start();
    #endif
    
    //? Compute B
    EMf->calculateB();
	
    #ifdef __PROFILING__
    time_b.stop();
    #endif

    #ifdef __PROFILING__
    time_div.start();
    #endif

    //? Compute divergences of E and B
    EMf->timeAveragedDivE(col->getPoissonMAdiv());
    EMf->divergence_E(col->getPoissonMAres());          //* Used to compute residual divergence for charge conservation
    EMf->divergence_B();

    #ifdef __PROFILING__
    time_div.stop();
    #endif

    #ifdef __PROFILING__
    time_total.stop();

    if (MPIdata::get_rank() == 0)
    {
        cout << endl << "Profiling of FIELD SOLVER" << endl; 
        cout << "Compute electric field     : " << time_e.total()     << " s, fraction of time taken in FieldSolver(): " << time_e.total()/time_total.total() << endl;
        cout << "Compute magnetic field     : " << time_b.total()     << " s, fraction of time taken in FieldSolver(): " << time_b.total()/time_total.total() << endl;
        cout << "Compute divergence of B    : " << time_div.total()   << " s, fraction of time taken in FieldSolver(): " << time_div.total()/time_total.total() << endl;
        cout << "FieldSolver()              : " << time_total.total() << " s" << endl << endl;
    }
    #endif
}


//! Compute positions and velocities of particles
int c_Solver::cudaLauncherAsync(const int species)
{
    cudaSetDevice(cudaDeviceOnNode); // a must on multi-device node

    cudaEvent_t event1, event2;
    cudaErrChk(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));
    auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
    const size_t bytes = (size_t)moments130::NUM_CHANNELS * gridSize * sizeof(cudaMomentType);

    //! Particle Splitting
    // constexpr bool PARTICLE_SPLITTING = true;
    // if constexpr(PARTICLE_SPLITTING)
    // {
    //     if(pclsArrayHostPtr[species]->getNOP() < 0.95 * pclsArrayHostPtr[species]->getInitialNOP())
    //     {
    //         const uint32_t deltaPcl = pclsArrayHostPtr[species]->getInitialNOP() - pclsArrayHostPtr[species]->getNOP();

    //         if(deltaPcl < pclsArrayHostPtr[species]->getNOP())
    //         {
    //             std::cout << "Particle splitting basic myrank: "<< MPIdata::get_rank() << " species " << species <<" number particles: " << pclsArrayHostPtr[species]->getNOP() <<
    //                     " delta: " << deltaPcl <<std::endl;
    //             particleSplittingKernel<false><<<getGridSize((int)deltaPcl, 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
    //             pclsArrayHostPtr[species]->setNOE(pclsArrayHostPtr[species]->getInitialNOP());
    //             cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species]));
    //             //cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
    //         }
    //         else
    //         { 
    //             // in this case the final number of particles will be < pclsArrayHostPtr[species]->getInitialNOP() 
    //             // worst case scenario will be pclsArrayHostPtr[species]->getInitialNOP() - (pclsArrayHostPtr[species]->getNOP() - 1)
    //             const int splittingTimes = deltaPcl / pclsArrayHostPtr[species]->getNOP();
    //             std::cout << "Particle splitting multipleTimesKernel myrank: "<< MPIdata::get_rank() << " species " << species <<" number particles: " << pclsArrayHostPtr[species]->getNOP() <<
    //                     " delta: " << deltaPcl <<std::endl;
    //             particleSplittingKernel<true><<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
    //             pclsArrayHostPtr[species]->setNOE( (splittingTimes + 1) * pclsArrayHostPtr[species]->getNOP());
    //             cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species]));
                
    //             //std::cerr << "Particle control multiple time splitting not yet implemented "<<std::endl;
    //         } 
    //     }
    // }
    
    const int oldNOP = pclsArrayHostPtr[species]->getNOP();

    //! ============================= Call Particle Mover ============================= !//

    //? Wait until "Field data copy Host --> Device" (event 0) is finished
    cudaErrChk(cudaStreamWaitEvent(streams[species], event0, 0));        

    //! Kernel to compute new positions and velocities of the particles
    // moverKernel <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);

    //TODO: Implement RelSIM
    // if Relativistic
    // RelSIM_velocity_kernel <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
    // else
    ECSIM_velocity_kernel <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    ECSIM_RelSIM_position_kernel <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (moverParamCUDAPtr[species], grid3DCUDACUDAPtr);

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());
    
    //? Open boundary condition
    // if (moverParamHostPtr[species]->doOpenBC) 
    // {   // update the particle number, OBC append
    //     updatePclNumAfterMoverKernel<<<1, 1, 0, streams[species]>>>(moverParamCUDAPtr[species]);
    // }

    //? Mark "Particle Mover complete" as event 1
    cudaErrChk(cudaEventRecord(event1, streams[species]));


    //! ============================= Call Moment Gatherer ============================= !//

    //? Set all moments to 0
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[species], 0, bytes, streams[species+ns]));       
    
    //TODO PJD: Do we need all results for position and velocity? If so, remove additional stream.

    //? Mark "Moments set to zero" as event 2
    cudaErrChk(cudaEventRecord(event2, streams[species+ns]));

    //? Wait until "Moments set to zero" (event 2) is finished 
    cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0)); 
    
    //! Deposit moments from particles that remained in their respective MPI subdomain
    ECSIM_RelSIM_Moments_PreExchange <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (momentParamCUDAPtr[species], grid3DCUDACUDAPtr, fieldForPclCUDAPtr, momentsCUDAPtr[species]);
    
    // momentKernelStayed <<< getGridSize(oldNOP, 256), 256, 0, streams[species] >>> (momentParamCUDAPtr[species], grid3DCUDACUDAPtr, momentsCUDAPtr[species]);

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());


    //! ============================= BC-related for particles ============================= !//
    
    //? Wait until "Particle Mover complete" (event 1) is finished
    cudaErrChk(cudaStreamWaitEvent(streams[species+ns], event1, 0));

    //? Pull hashed exit/delete counters back to host
    cudaErrChk(cudaMemcpyAsync(hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species], departureArrayElementType::DELETE*sizeof(hashedSum), cudaMemcpyDefault, streams[species+ns]));
    
    //? Ensure counters are available on host
    cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
    
    //? Sync particle-array header (e.g., NOP) host<-device
    cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[species], pclsArrayCUDAPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species+ns]));

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    int x = 0; // exiting particle number
    for(int i=0; i<departureArrayElementType::DELETE_HASHEDSUM_INDEX; i++)
        x += hashedSumArrayHostPtr[species][i].getSum();
    
    const int y = hashedSumArrayHostPtr[species][departureArrayElementType::DELETE_HASHEDSUM_INDEX].getSum(); // deleted particle number
    const int hole = x + y;

    if(x > exitingArrayHostPtr[species]->getSize())
    { 
        // prepare the exitingArray
        exitingArrayHostPtr[species]->expand(x * 1.5, streams[species+ns]);
        cudaErrChk(cudaMemcpyAsync(exitingArrayCUDAPtr[species], exitingArrayHostPtr[species], sizeof(exitingArray), cudaMemcpyDefault, streams[species+ns]));
    }

    if(hole > fillerBufferArrayHostPtr[species]->getSize())
    {
        // prepare the fillerBuffer
        fillerBufferArrayHostPtr[species]->expand(hole * 1.5, streams[species+ns]);
        cudaErrChk(cudaMemcpyAsync(fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species], sizeof(fillerBuffer), cudaMemcpyDefault, streams[species+ns]));
    }

    if(x > particles[species].get_pcl_list().capacity())
    {
        // expand the host array
        auto pclArray = particles[species].get_pcl_arrayPtr();
        pclArray->reserve(x * 1.5);
    }

    exitingKernel <<< getGridSize(oldNOP / 4, 256), 256, 0, streams[species+ns] >>> (pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species], hole);
    cudaErrChk(cudaEventRecord(event2, streams[species+ns]));
    
    // Copy exiting particle to host
    cudaErrChk(cudaMemcpyAsync(particles[species].get_pcl_array().getList(), exitingArrayHostPtr[species]->getArray(), x*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[species+ns]));
    particles[species].get_pcl_array().setSize(x);

    // Sorting
    cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0));
    if (hole > 0)
    {
        sortingKernel1 <<< getGridSize(hole, 128), 128, 0, streams[species] >>> (pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
        fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::FILLER_HASHEDSUM_INDEX, hole);
        sortingKernel2 <<< getGridSize((oldNOP-hole)/4, 256), 256, 0, streams[species] >>> (pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
        fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::HOLE_HASHEDSUM_INDEX, hole);
    }

    cudaErrChk(cudaEventDestroy(event1));
    cudaErrChk(cudaEventDestroy(event2));
    cudaErrChk(cudaStreamSynchronize(streams[species+ns])); // exiting particle copied

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    return hole; // Number of exiting + deleted particles
}

//TODO: The particles of a single species (the one with index mergeIdx) are being merged—i.e., 
//TODO: multiple nearby macroparticles in that species are coalesced into fewer “superparticles” inside each cell to control particle count/noise.

bool c_Solver::ParticlesMoverMomentAsync()
{
    //? Get the field data
    EMf->set_fieldForPclsToCenter(fieldForPclHostPtr);

    // cout << "set_fieldForPclsToCenter done" << endl;

    //? Asynchronously copy the field data from Host --> Device
    cudaErrChk(cudaMemcpyAsync(fieldForPclCUDAPtr, fieldForPclHostPtr, (grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1)) * 24 * sizeof(cudaFieldType), cudaMemcpyDefault, streams[0]));
    //TODO PJD: 24 is a packed component count (e.g., 3 fields × 8 stencil points or similar).
    
    //? Mark "Field data copy Host --> Device" as event0
    cudaErrChk(cudaEventRecord(event0, streams[0]));

    //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    //TODO: Is merging done in the CPU code? - PJD

    //* Iterate over all species
    for(int species = 0; species < ns; species++)
    {
        //? Apply cudaLauncherAsync on each species asynchronously (except the one being merged)
        // if (species != mergeIdx)
        // {
            exitingResults[species] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, species);

            // toBeMerged[2 * species + 1] +=1;
        // }
    }

    // if (mergeIdx >= 0 && mergeIdx < ns)
    // {
    //     const auto& i = mergeIdx;
    //     std::cout << " Particle merging myrank: "<< MPIdata::get_rank() << " species: " << i << std::endl;

    //     cudaErrChk(cudaStreamSynchronize(streams[i])); // wait for the copy
    //     // sort
    //     outputPart[i].sort_particles_parallel(cellCountHostPtr, cellOffsetHostPtr);

    //     const int totalCells = grid->getNXC() * grid->getNYC() * grid->getNZC();
    //     cudaErrChk(cudaMemcpyAsync(cellCountCUDAPtr, cellCountHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));
    //     cudaErrChk(cudaMemcpyAsync(cellOffsetCUDAPtr, cellOffsetHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));

    //     cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getpcls(), outputPart[i].get_pcl_array().getList(), 
    //                             pclsArrayHostPtr[i]->getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[i]));

    //     // merge
    //     mergingKernel<<<getGridSize(totalCells * WARP_SIZE, 256), 256, 0, streams[i]>>>(cellOffsetCUDAPtr, cellCountCUDAPtr, grid3DCUDACUDAPtr, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);

    //     exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i);

    //     toBeMerged[2 * i + 1] = 0;
    //     mergeIdx = -1; // merged
    // }

    return (false);
}

bool c_Solver::MoverAwaitAndPclExchange()
{

    for (int i = 0; i < ns; i++)
    { 
        auto x = exitingResults[i].get(); // holes
        stayedParticle[i] = pclsArrayHostPtr[i]->getNOP() - x;
    }

        //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    // exiting particles are copied back

    for (int i = 0; i < ns; i++)  // communicate each species
    {
        auto a = particles[i].separate_and_send_particles();
        particles[i].recommunicate_particles_until_done(1);
        
        //? Particle Injection
        // if (moverParamHostPtr[i]->doRepopulateInjection) 
        // { // Now particles contains incoming particles and the injection
        //     particles[i].repopulate_particles_onlyInjection();
        // }
    }

        //! Remove - PJD
    cudaErrChk(cudaDeviceSynchronize());

    for(int i=0; i<ns; i++)
    {
        // Copy repopulate particle and the incoming particles to device
        pclsArrayHostPtr[i]->setNOE(stayedParticle[i]); // After the Sorting
        auto newPclNum = stayedParticle[i] + particles[i].getNOP();

        // now the host array contains the entering particles
        if((newPclNum * 1.2) >= pclsArrayHostPtr[i]->getSize())
        {   
            // not enough size, expand the device array size
            pclsArrayHostPtr[i]->expand(newPclNum * 1.5, streams[i]);
            departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
            cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i], sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
        }

        // now enough size on device pcls array, copy particles
        cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getpcls() + pclsArrayHostPtr[i]->getNOP(), (void*)&(particles[i].get_pcl_list()[0]), particles[i].getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[i]));
                
        pclsArrayHostPtr[i]->setNOE(newPclNum); 
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));    
        
        //! Remove - PJD
        cudaErrChk(cudaDeviceSynchronize());

        //! Deposit moments from particles entering new MPI subdomain (after particle exchange)
        if(pclsArrayHostPtr[i]->getNOP() - stayedParticle[i] > 0)
            ECSIM_RelSIM_Moments_PostExchange <<< getGridSize(pclsArrayHostPtr[i]->getNOP() - stayedParticle[i], 128u), 128, 0, streams[i] >>> (momentParamCUDAPtr[i], grid3DCUDACUDAPtr, fieldForPclCUDAPtr, momentsCUDAPtr[i], stayedParticle[i]);

        //! Remove - PJD
        cudaErrChk(cudaDeviceSynchronize());

        // reset the hashedSum, and the departureArray, which is a must
        for(int j=0; j<departureArrayElementType::HASHED_SUM_NUM; j++)
            hashedSumArrayHostPtr[i][j].resetBucket();
        
        cudaErrChk(cudaMemcpyAsync(hashedSumArrayCUDAPtr[i], hashedSumArrayHostPtr[i], (departureArrayElementType::HASHED_SUM_NUM) * sizeof(hashedSum), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType), streams[i]));
    }

    //? --------------------- Copy moments from Device to Host --------------------- ?//

    auto oneDensity = (uint64_t)grid->getNXN() * (uint64_t)grid->getNYN() * (uint64_t)grid->getNZN();
    
    // for(int i=0; i<ns; i++)
    // {
    //     cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)),  momentsCUDAPtr[i]+0*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //     cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxhs().get(i,0,0,0)),   momentsCUDAPtr[i]+1*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //     cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJyhs().get(i,0,0,0)),   momentsCUDAPtr[i]+2*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //     cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzhs().get(i,0,0,0)),   momentsCUDAPtr[i]+3*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
 
    //     for (int ind = 0; ind < 14; ++ind) 
    //     {
    //         const int baseCh = mass_base + ind * comps_per_neighbor;

    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxx().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 0)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxy().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 1)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxz().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 2)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyx().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 3)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyy().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 4)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyz().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 5)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzx().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 6)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzy().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 7)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //         cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzz().get(ind,0,0,0)), momentsCUDAPtr[i] + (baseCh + 8)*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    //     }
    // }

    for(int i=0; i<ns; i++)
    {
        //* Copy moments back to field object EMf (Device --> Host)
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_RHO, oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JX, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJyhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JY, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzhs().get(i,0,0,0)),  momentsCUDAPtr[i] + moments130::chan_offset(moments130::CH_JZ, oneDensity),  oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
 
        for (int ind = 0; ind < moments130::NUM_NEIGHBORS; ++ind)
        {
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMxz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 0, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMyz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 1, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzx().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 0), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzy().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 1), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
            cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getMzz().get(ind,0,0,0)), momentsCUDAPtr[i] + moments130::chan_offset(moments130::mm_channel(ind, 2, 2), oneDensity), oneDensity * sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
        }
    }

    return (false);
}


//! ===================================== WRITE DATA TO FILES ===================================== !//

void c_Solver::writeParticleNum(int cycle) 
{
    pclNumCSV << cycle << ",";

    for(int i=0; i<ns-1; i++)
        pclNumCSV << pclsArrayHostPtr[i]->getNOP() << ",";
    
    pclNumCSV << pclsArrayHostPtr[ns-1]->getNOP() << std::endl;
}

void c_Solver::WriteOutput(int cycle) 
{
    #ifdef USE_CATALYST
    Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
    #endif

    WriteConserved(cycle);
    WriteRestart(cycle);


    if(!Parameters::get_doWriteOutput())  return;

    if (col->getWriteMethod() == "nbcvtk"){//Non-blocking collective MPI-IO

        if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
            if(!(col->getFieldOutputTag()).empty()){

                if(fieldreqcounter>0){
                        
                        //MPI_Waitall(fieldreqcounter,&fieldreqArr[0],&fieldstsArr[0]);
                    for(int si=0;si< fieldreqcounter;si++){
                        int error_code = MPI_File_write_all_end(fieldfhArr[si],&fieldwritebuffer[si][0][0][0],&fieldstsArr[si]);//fieldstsArr[si].MPI_ERROR;
                        if (error_code != MPI_SUCCESS) {
                            char error_string[100];
                            int length_of_error_string, error_class;
                            MPI_Error_class(error_code, &error_class);
                            MPI_Error_string(error_class, error_string, &length_of_error_string);
                            dprintf("MPI_Waitall error at field output cycle %d  %d  %s\n",cycle, si, error_string);
                        }else{
                            MPI_File_close(&(fieldfhArr[si]));
                        }
                    }
                }
                fieldreqcounter = WriteFieldsVTKNonblk(grid, EMf, col, vct,cycle,fieldwritebuffer,fieldreqArr,fieldfhArr);
            }

            if(!(col->getMomentsOutputTag()).empty()){

                if(momentreqcounter>0){
                    //MPI_Waitall(momentreqcounter,&momentreqArr[0],&momentstsArr[0]);
                    for(int si=0;si< momentreqcounter;si++){
                        int error_code = MPI_File_write_all_end(momentfhArr[si],&momentwritebuffer[si][0][0],&momentstsArr[si]);//momentstsArr[si].MPI_ERROR;
                        if (error_code != MPI_SUCCESS) {
                            char error_string[100];
                            int length_of_error_string, error_class;
                            MPI_Error_class(error_code, &error_class);
                            MPI_Error_string(error_class, error_string, &length_of_error_string);
                            dprintf("MPI_Waitall error at moments output cycle %d  %d %s\n",cycle, si, error_string);
                        }else{
                            MPI_File_close(&(momentfhArr[si]));
                        }
                    }
                }
                momentreqcounter = WriteMomentsVTKNonblk(grid, EMf, col, vct,cycle,momentwritebuffer,momentreqArr,momentfhArr);
            }
        }

            WriteParticles(cycle);
            // WriteTestParticles(cycle);

    }else if (col->getWriteMethod() == "pvtk"){//Blocking collective MPI-IO
        if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
            if(!(col->getFieldOutputTag()).empty()){
                //WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle);//B + E + Je + Ji + rho
                WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle, fieldwritebuffer);//B + E + Je + Ji + rho
            }
            if(!(col->getMomentsOutputTag()).empty()){
                WriteMomentsVTK(grid, EMf, col, vct, col->getMomentsOutputTag() ,cycle, momentwritebuffer);
            }
        }

        WriteParticles(cycle);
            // WriteTestParticles(cycle);

    } else{
        throw std::runtime_error("Unknown output method: " + col->getWriteMethod());
    }
}

//! Asynchronously stage particle arrays for restart or particle output
void c_Solver::outputCopyAsync(int cycle)
{ 
    const auto ifNextCycleRestart = restart_cycle>0 && (cycle+1)%restart_cycle==0;                                      //* Restart
    const auto ifNextCycleParticle = !col->particle_output_is_off() && (cycle+1)%(col->getParticlesOutputCycle())==0;   //* Particle output
    
    //* Prepare copies if particle or restart output is due on the next step
    if (ifNextCycleRestart || ifNextCycleParticle)
    {
        for(int i=0; i<ns; i++)
        {
            if(outputPart[i].get_pcl_array().capacity() < pclsArrayHostPtr[i]->getNOP())                                //* Ensure staging buffer has enough capacity
            {
                // expand the host array
                auto pclArray = outputPart[i].get_pcl_arrayPtr();                                                       //* Get modifiable handle to the AoS particle array
                pclArray->reserve(pclsArrayHostPtr[i]->getNOP() * 1.2);                                                 //* Reserve extra space to reduce future resizes
            }

            cudaErrChk(cudaMemcpyAsync(outputPart[i].get_pcl_array().getList(), pclsArrayHostPtr[i]->getpcls(),         //* Async copy current host particle AoS to staging buffer (Host --> Host)
                                       pclsArrayHostPtr[i]->getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[0]));
            
            outputPart[i].get_pcl_array().setSize(pclsArrayHostPtr[i]->getNOP());                                       //* Set the number of items now in the staging array
        }

        cudaErrChk(cudaEventRecord(eventOutputCopy, streams[0]));                                                        //* Record an event to signal "output copy finished" on stream 0
    }
}

void c_Solver::WriteRestart(int cycle)
{
  if (restart_cycle>0 && cycle%restart_cycle==0){

    cudaErrChk(cudaEventSynchronize(eventOutputCopy));

	  convertOutputParticlesToSynched();

#ifdef USE_ADIOS2
    adiosManager->appendRestartOutput(cycle);
#endif
  }
}

// write the conserved quantities
void c_Solver::WriteConserved(int cycle) 
{
    if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
    {
        Eenergy = EMf->get_E_field_energy();
        Benergy = EMf->get_B_field_energy();
        TOTenergy = 0.0;
        TOTmomentum = 0.0;
        
        for (int is = 0; is < ns; is++) 
        {
            kinetic_energy_species[is] = outputPart[is].get_kinetic_energy();
            bulk_energy_species[is] = EMf->get_bulk_energy(is);
            TOTenergy += kinetic_energy_species[is];
            momentum_species[is] = outputPart[is].get_momentum();
            TOTmomentum += momentum_species[is];
        }
        
        if (myrank == (nprocs-1)) 
        {
            ofstream my_file(cq.c_str(), fstream::app);
            if(cycle == 0)my_file << "\t" << "\t" << "\t" << "Total_Energy" << "\t" << "Momentum" << "\t" << "Eenergy" << "\t" << "Benergy" << "\t" << "Kenergy" << "\t" << "Kenergy(species)" << "\t" << "bulk_energy_species(species)" << endl;
            my_file << cycle << "\t" << "\t" << (Eenergy + Benergy + TOTenergy) << "\t" << TOTmomentum << "\t" << Eenergy << "\t" << Benergy << "\t" << TOTenergy;
            for (int is = 0; is < ns; is++) my_file << "\t" << kinetic_energy_species[is];
            for (int is = 0; is < ns; is++) my_file << "\t" << bulk_energy_species[is];
            my_file << endl;
            my_file.close();
        }
    }
}

void c_Solver::WriteVelocityDistribution(int cycle)
{
  // Velocity distribution
  //if(cycle % col->getVelocityDistributionOutputCycle() == 0)
  {
    for (int is = 0; is < ns; is++) {
      double maxVel = outputPart[is].getMaxVelocity();
      double* VelocityDist = outputPart[is].getVelocityDistribution(nDistributionBins, 0, maxVel);
      if (myrank == 0) {
        ofstream my_file(ds.c_str(), fstream::app);
        my_file << cycle << "\t" << is << "\t" << maxVel;
        for (int i = 0; i < nDistributionBins; i++)
          my_file << "\t" << VelocityDist[i];
        my_file << endl;
        my_file.close();
      }
      delete [] VelocityDist;
    }
  }
}

// This seems to record values at a grid of sample points
//
void c_Solver::WriteVirtualSatelliteTraces()
{
  if(ns <= 2) return;
  assert_eq(ns,4);

  ofstream my_file(cqsat.c_str(), fstream::app);
  const int nx0 = grid->get_nxc_r();
  const int ny0 = grid->get_nyc_r();
  const int nz0 = grid->get_nzc_r();
  for (int isat = 0; isat < nsat; isat++) {
    for (int jsat = 0; jsat < nsat; jsat++) {
      for (int ksat = 0; ksat < nsat; ksat++) {
        int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
        int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
        int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
        my_file << EMf->getBx(index1, index2, index3) << "\t" << EMf->getBy(index1, index2, index3) << "\t" << EMf->getBz(index1, index2, index3) << "\t";
        my_file << EMf->getEx(index1, index2, index3) << "\t" << EMf->getEy(index1, index2, index3) << "\t" << EMf->getEz(index1, index2, index3) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 0) + EMf->getJxs(index1, index2, index3, 2) << "\t" << EMf->getJys(index1, index2, index3, 0) + EMf->getJys(index1, index2, index3, 2) << "\t" << EMf->getJzs(index1, index2, index3, 0) + EMf->getJzs(index1, index2, index3, 2) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 1) + EMf->getJxs(index1, index2, index3, 3) << "\t" << EMf->getJys(index1, index2, index3, 1) + EMf->getJys(index1, index2, index3, 3) << "\t" << EMf->getJzs(index1, index2, index3, 1) + EMf->getJzs(index1, index2, index3, 3) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 0) + EMf->getRHOns(index1, index2, index3, 2) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 1) + EMf->getRHOns(index1, index2, index3, 3) << "\t";
      }}}
  my_file << endl;
  my_file.close();
}

// void c_Solver::WriteFields(int cycle) = delete;

void c_Solver::WriteParticles(int cycle)
{
  if(col->particle_output_is_off() || cycle%(col->getParticlesOutputCycle())!=0) return;

  cudaErrChk(cudaEventSynchronize(eventOutputCopy));

  // this is a hack
  for (int i = 0; i < ns; i++){
    outputPart[i].set_particleType(ParticleType::Type::AoS); // this is a even more hack
    outputPart[i].convertParticlesToSynched();
  }

#ifdef USE_ADIOS2
  adiosManager->appendParticleOutput(cycle);
#endif

}

//! =============================================================================================== !//

void c_Solver::Finalize() 
{
	pclNumCSV.close();
	
	if (col->getCallFinalize() && Parameters::get_doWriteOutput() && col->getRestartOutputCycle() >= 0)
	{
		outputCopyAsync(-1);
		cudaErrChk(cudaEventSynchronize(eventOutputCopy));

		convertOutputParticlesToSynched();

		#ifdef USE_ADIOS2
			adiosManager->appendRestartOutput((col->getNcycles() + first_cycle) - 1);
		#endif
	}

	#ifdef USE_ADIOS2
		adiosManager->closeOutputFiles();
	#endif

	deInitCUDA();

	my_clock->stopTiming();
}

//! Place the particles into new cells according to their current position
void c_Solver::sortParticles() 
{
	for(int species_idx=0; species_idx<ns; species_idx++)
		particles[species_idx].sort_particles_serial();
}

void c_Solver::pad_particle_capacities()
{
	for (int i = 0; i < ns; i++)
		particles[i].pad_capacities();

	for (int i = 0; i < nstestpart; i++)
		testpart[i].pad_capacities();
}

//? Convert particle data to struct of arrays (assumed by I/O)
void c_Solver::convertParticlesToSoA()
{
	for (int i = 0; i < ns; i++)
		particles[i].convertParticlesToSoA();
}

//? Convert particle data to array of structs (used in computing)
void c_Solver::convertParticlesToAoS()
{
	for (int i = 0; i < ns; i++)
		particles[i].convertParticlesToAoS();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertOutputParticlesToSynched()
{
	for (int i = 0; i < ns; i++)
	{
		outputPart[i].set_particleType(ParticleType::Type::AoS); // otherwise the output is not synched
		outputPart[i].convertParticlesToSynched();
	}

	for (int i = 0; i < nstestpart; i++)
	{
		testpart[i].set_particleType(ParticleType::Type::AoS); // otherwise the output is not synched
		testpart[i].convertParticlesToSynched();
	}
}

int c_Solver::LastCycle() 
{
    return (col->getNcycles() + first_cycle);
}
