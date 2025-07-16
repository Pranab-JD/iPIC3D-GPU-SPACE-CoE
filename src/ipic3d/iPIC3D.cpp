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


#include "MPIdata.h"
#include "iPic3D.h"
#include "debug.h"
#include "TimeTasks.h"
#include <stdio.h>
#include <chrono>

#include "dataAnalysis.cuh"
#include "utility/LeXInt_Timer.hpp"

using namespace iPic3D;

int main(int argc, char **argv) 
{
    //* Initialise MPI
    MPIdata::init(&argc, &argv);
    
    {
        iPic3D::c_Solver KCode;
        KCode.Init(argc, argv); //! load param from file, init the grid, fields
        dataAnalysis::dataAnalysisPipeline DA(KCode); // has to be created after KCode.Init()

        //? LeXInt timer
        LeXInt::timer time_EF, time_PM, time_MF, time_MG, time_WD, time_EX, time_CP;

        //? Initial moment computation
        KCode.CalculateMoments();

        for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) 
        {
            if (KCode.get_myrank() == 0)
                std::cout << std::endl << "=================== Cycle " << i << " ===================" << std::endl ;
            
            timeTasks.resetCycle();

            KCode.writeParticleNum(i);

            //? Field Solver --> Compute E field 
            DA.startAnalysis(i);
            time_EF.start();
            KCode.CalculateField(i); // E field, spare GPU cycles
            time_EF.stop();
            DA.waitForAnalysis();

            //? Particle Pusher --> Compute new velocities and positions of the particles
            time_PM.start();
            KCode.ParticlesMoverMomentAsync(); // launch Mover and Moment kernels
            time_PM.stop();

            time_WD.start();
            // KCode.WriteOutput(i);    // some spare CPU cycles
            time_WD.stop();

            time_EX.start();
            KCode.MoverAwaitAndPclExchange();
            time_EX.stop();

            //? Field Solver --> Compute B fields 
            time_MF.start();
            KCode.CalculateB();     // some spare CPU cycles
            time_MF.stop();

            //? Moment Gatherer --> Compute charge density, current density, and pressure tnesor
            time_MG.start();
            KCode.MomentsAwait();
            time_MG.stop();

            time_CP.start();
            KCode.outputCopyAsync(i); // copy output data to host, for next output
            time_CP.stop();

            if(MPIdata::get_rank() == 0)
            {
                std::cout << std::endl << "Runtime of iPIC3D modules " << std::endl;
                std::cout << "Field solver (E)   : " << time_EF.total()   << " s" << std::endl;
                std::cout << "Field solver (B)   : " << time_MF.total()   << " s" << std::endl;
                std::cout << "Particle mover     : " << time_PM.total()   << " s" << std::endl;
                std::cout << "Moment gatherer    : " << time_MG.total()   << " s" << std::endl;
            }

            if(MPIdata::get_rank() == 0)
            {
                std::cout << std::endl << "Runtime of other core functions " << std::endl;
                std::cout << "Write data         : " << time_WD.total()   << " s" << std::endl;
                std::cout << "Exchange Particles : " << time_EX.total()   << " s" << std::endl;
                std::cout << "Copy data          : " << time_CP.total()   << " s" << std::endl;
            }
        }

        KCode.Finalize();
    }

    //* Finalise MPI
    MPIdata::instance().finalize_mpi();

    return 0;
}
