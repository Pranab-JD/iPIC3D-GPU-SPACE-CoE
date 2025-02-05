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

#include "performances/LeXInt_Timer.hpp"

using namespace iPic3D;

int main(int argc, char **argv) 
{

    MPIdata::init(&argc, &argv);
    {

        iPic3D::c_Solver KCode;
        
        KCode.Init(argc, argv); //! load param from file, init the grid, fields

        timeTasks.resetCycle(); //reset timer

        //? LeXInt timer
        LeXInt::timer time_EF, time_PM, time_MF, time_MG, time_loop, time_total;
        time_total.start();
  
        //? Initial Moment Gatherer
        KCode.CalculateMoments(true);

        for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++)
        {
            if (KCode.get_myrank() == 0)
                std::cout << std::endl << "=================== Cycle " << i << " ===================" << std::endl ;

            timeTasks.resetCycle();
            
            //? Field Solver --> Compute E field 
            time_EF.start();
            KCode.CalculateField(i);
            time_EF.stop();
            
            //? Particle Pusher --> Compute new velocities and positions of the particles
            time_PM.start();
            KCode.ParticlesMover();
            time_PM.stop();
            
            //? Field Solver --> Compute B field
            time_MF.start();
            KCode.CalculateB();
            time_MF.stop();
            
            //? Moment Gatherer --> Compute charge density, current density, and pressure tensor
            time_MG.start();
            KCode.CalculateMoments(false);
            time_MG.stop();

            KCode.WriteOutput(i);
            
            //? Print out total time for all tasks
            #ifdef LOG_TASKS_TOTAL_TIME
                timeTasks.print_cycle_times(i);
            #endif

            if(MPIdata::get_rank() == 0)
            {
                std::cout << std::endl << "Runtime of iPIC3D modules " << std::endl;
                std::cout << "Field solver (E)   : " << time_EF.total()   << " s" << std::endl;
                std::cout << "Field solver (B)   : " << time_MF.total()   << " s" << std::endl;
                std::cout << "Particle mover     : " << time_PM.total()   << " s" << std::endl;
                std::cout << "Moment gatherer    : " << time_MG.total()   << " s" << std::endl;
            }
        }

        #ifdef LOG_TASKS_TOTAL_TIME
            timeTasks.print_tasks_total_times();
        #endif

        KCode.Finalize();
    }

    //? close MPI
    MPIdata::instance().finalize_mpi();

    return 0;
}
