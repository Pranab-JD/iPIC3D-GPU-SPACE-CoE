#ifdef USE_ADIOS2

#include "ADIOS2IO.hpp"
#include "iPic3D.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
#include "Collective.h"

#include "mpi.h"
#include "adios2.h"

#include "MPIdata.h"

#include <chrono>

namespace ADIOS2IO {

using namespace std;


void ADIOS2Manager::initOutputFiles(string fieldTag, string particleTag, int sample, iPic3D::c_Solver& KCode) {

    if (open) {
        closeOutputFiles();
    }

    this->cartisianRank = KCode.vct->getCartesian_rank();
    this->saveDirName = KCode.col->getSaveDirName();
    this->restartDirName = KCode.col->getRestartDirName();
    this->restartStatus = KCode.col->getRestart_status();

    this->fieldTag = fieldTag;
    this->particleTag = particleTag;
    this->sample = sample;

    this->col = KCode.col;
    this->vct = KCode.vct;
    this->grid = KCode.grid;
    this->EMf = KCode.EMf;

    this->part = KCode.outputPart;
    this->ns = KCode.col->getNs();
    this->testpart = KCode.testpart;
    this->nstestpart = KCode.col->getNsTestPart();


    // ADIOS2
    this->adios = adios2::ADIOS(MPIdata::get_PicGlobalComm());

    // open files
    if (!fieldTag.empty()) { throw runtime_error("Field output is not supported yet"); 
        this->ioField = adios.DeclareIO("FieldOutput");
        this->ioField.SetEngine("BP5");
        auto filePath = saveDirName + "/field_" + to_string(cartisianRank) + ".bp";
        engineField = ioField.Open(filePath, adios2::Mode::Write);

    }

    if (!particleTag.empty()) {
        this->ioParticle = adios.DeclareIO("ParticleOutput");
        this->ioParticle.SetEngine("BP5");
        auto filePath = saveDirName + "/particle_" + to_string(cartisianRank) + ".bp";
        engineParticle = ioParticle.Open(filePath, adios2::Mode::Write, MPI_COMM_SELF);

        // parse the tag and prepae the map
        particleTag.erase(remove(particleTag.begin(), particleTag.end(), ' '), particleTag.end());
        vector<string> tags;
        stringstream ss(particleTag);
        string tag;
        while (getline(ss, tag, '+')) {
            tags.push_back(tag);
        }

        // find the function in the map and register it to vector
        for (auto tag : tags) {
            if (outputTagOptions.find(tag) != outputTagOptions.end()) {
                particleOptions.push_back(outputTagOptions[tag]);
            } else {
                throw runtime_error("Particle output tag is not supported: " + tag);
            }
        }


    }

    if (restartStatus > 0) { throw runtime_error("Restart output is not supported yet"); 
        this->ioRestart = adios.DeclareIO("RestartOutput");
        this->ioRestart.SetEngine("BP5");
        auto filePath = restartDirName + "/restart_" + to_string(cartisianRank) + ".bp";
        engineRestart = ioRestart.Open(filePath, adios2::Mode::Write, MPI_COMM_SELF);


    }


    open = true;

}



void ADIOS2Manager::appendFieldOutput(int cycle) {
    throw runtime_error("Field output is not supported yet");
}

void ADIOS2Manager::appendParticleOutput(int cycle) {
    if (particleOptions.empty()) return;

    engineParticle.BeginStep();

    auto cycleVar = _variableHelper<int>(ioParticle, "cycle");
    engineParticle.Put<int>(cycleVar, cycle);

    auto timeVar = _variableHelper<int>(ioParticle, "IOTimeMS");
    auto start = chrono::high_resolution_clock::now();

    for (auto option : particleOptions) {
        option(ioParticle, engineParticle);
    }

    engineParticle.PerformPuts(); // do the heavy job here

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    engineParticle.Put<int>(timeVar, duration.count());

    engineParticle.EndStep();

}


void ADIOS2Manager::appendRestartOutput(int cycle) {
    throw runtime_error("Restart output is not supported yet");
}


void ADIOS2Manager::appendOutput(int cycle) {
    if (!open) throw runtime_error("Output files are not open");


    appendParticleOutput(cycle);

    outputCount++;
    lastCycle = cycle;

}


void ADIOS2Manager::closeOutputFiles() {

    if (!open) return;

    if (!fieldTag.empty()) {
        engineField.Close();
    }

    if (!particleTag.empty()) {
        engineParticle.Close();
    }

    if (restartStatus > 0) {
        engineRestart.Close();
    }

    open = false;
}







} // end namespace ADIOS2IO

#endif