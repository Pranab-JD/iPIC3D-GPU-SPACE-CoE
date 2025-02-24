#ifndef _ADIOS2_HPP_
#define _ADIOS2_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "ipicfwd.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "iPic3D.h"

#include "adios2.h"

namespace iPic3D {
    class c_Solver;
}

namespace ADIOS2IO {

using namespace std;

class ADIOS2Manager {

private:
    // adios2
    adios2::ADIOS adios;

    adios2::IO ioField;
    adios2::Engine engineField;
    vector <function<void(int)>> fieldOptions;

    adios2::IO ioParticle;
    adios2::Engine engineParticle;
    vector <function<void(int)>> particleOptions;
    
    adios2::IO ioRestart;
    adios2::Engine engineRestart;
    vector <function<void(int)>> restartOptions;

    unordered_map<string, function<void(int)>> outputTagOptions = {
        {"position", bind(&ADIOS2Manager::_particlePosition, this, placeholders::_1)},
        {"velocity", bind(&ADIOS2Manager::_particleVelocity, this, placeholders::_1)},
        {"q", bind(&ADIOS2Manager::_particleCharge, this, placeholders::_1)},
        {"ID", bind(&ADIOS2Manager::_particleID, this, placeholders::_1)}
    };

    // general
    int cartisianRank;
    string saveDirName;
    string restartDirName;

    bool open = false; // open flag
    int lastCycle = -1; // last output simulation cycle
    int outputCount = 0; // output times count

    // field
    string fieldFile;
    string fieldTag;

    // particle
    string particleFile;
    string particleTag;
    int sample;

    // restart
    int restartStatus; //! restart_status 0 --> no restart; 1--> restart, create new; 2--> restart, append;
    string restartFile;


    // pointer registration
    Collective *col;
    VCtopology3D *vct;
    Grid3DCU *grid;
    EMfields3D *EMf;
    Particles3D *part; // now we only copy from the CPU buffer
    // particleArrayCUDA **pclsArrayHostPtr;
    Particles3D *testpart;
    int ns;
    int nstestpart;
    

public:

/**
 * @brief Construct a new ADIOS2Manager object
 * 
 * @details create or open the output files, register the pointers, configure the output routine
 */
void initOutputFiles(string fieldTag, string particleTag, int sample, iPic3D::c_Solver& KCode);

/**
 * @brief Append the output data to the output files, the interface 
 * 
 * @param cycle simulation cycle
 */
void appendOutput(int cycle);

void closeOutputFiles();

// void loadRestart(iPic3D::c_Solver& KCode);

private:
/**
 * @brief these are the output routines for different categories of data
 */

void appendFieldOutput(int cycle); 

void appendParticleOutput(int cycle);

void appendRestartOutput(int cycle);

/**
 * @brief helper function to create or inquire a variable
 */
template < typename T >
adios2::Variable<T> _variableHelper(adios2::IO &io, const std::string &name, const adios2::Dims &shape = adios2::Dims(), 
                                    const adios2::Dims &start = adios2::Dims(), const adios2::Dims &count = adios2::Dims(),
                                    const bool constantDims = false) {
    
    auto var = io.InquireVariable<T>(name);

    if(var){ // variable exists
        if (!shape.empty()) { // is array
            var.SetShape(shape);
            var.SetSelection({start, count});
        }
    } 
    else { // first time define
        var = shape.empty() ? io.DefineVariable<T>(name) : io.DefineVariable<T>(name, shape, start, count, constantDims);
        if (!var) throw std::runtime_error("Failed to define variable: " + name);
        
        // Operator
        if (name.find("cycle") == string::npos) {
            // auto ZFPOp = io.DefineOperator("CompressorZFP", adios2::ops::LossyZFP);
            var.AddOperation(adios2::ops::LossyZFP, {{"rate", "32"}, {"backend", "omp"}});
            
        }
    }

    return var;            
}

// tag mapping
/* Field
    collective
    total_topology 
    proc_topology
    Ball --> to write all B components
    Bx,By,Bz
    Eall --> to write all E components
    Ex,Ey,Ez
    phi --> scalar vector
    Jall --> to write all J (current density) components
    Jx,Jy,Jz
    Jsall --> to write all Js (current densities for each species) components
    Jxs,Jys,Jzs
    rho -> net charge density
    rhos -> charge densities for each species
    pressure -> pressure tensor for each species
    k_energy -> kinetic energy for each species
    B_energy -> energy of magnetic field
    E_energy -> energy of electric field
*/


/* Particle
    position -> particle position (x,y)
    velocity -> particle velocity (u,v,w)
    q -> particle charge
    ID -> particle ID (note: TrackParticleID has to be set true in Collective)
*/
void _particlePosition(int cycle){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto x = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "PositionX", {sizeNOP}, {0}, {sizeNOP});
        auto y = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "PositionY", {sizeNOP}, {0}, {sizeNOP});
        auto z = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "PositionZ", {sizeNOP}, {0}, {sizeNOP});

        engineParticle.Put<cudaCommonType>(x, part[i].getXall(), adios2::Mode::Deferred);
        engineParticle.Put<cudaCommonType>(y, part[i].getYall(), adios2::Mode::Deferred);
        engineParticle.Put<cudaCommonType>(z, part[i].getZall(), adios2::Mode::Deferred);
    }
}

void _particleVelocity(int cycle){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto u = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "VelocityU", {sizeNOP}, {0}, {sizeNOP});
        auto v = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "VelocityV", {sizeNOP}, {0}, {sizeNOP});
        auto w = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "VelocityW", {sizeNOP}, {0}, {sizeNOP});

        engineParticle.Put<cudaCommonType>(u, part[i].getUall(), adios2::Mode::Deferred);
        engineParticle.Put<cudaCommonType>(v, part[i].getVall(), adios2::Mode::Deferred);
        engineParticle.Put<cudaCommonType>(w, part[i].getWall(), adios2::Mode::Deferred);
    }
}

void _particleCharge(int cycle){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto var = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "charge", {sizeNOP}, {0}, {sizeNOP});

        engineParticle.Put<cudaCommonType>(var, part[i].getQall(), adios2::Mode::Deferred);
    }
}

void _particleID(int cycle){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto var = _variableHelper<cudaCommonType>(ioParticle, "part" + to_string(i) + "ID", {sizeNOP}, {0}, {sizeNOP});

        engineParticle.Put<cudaCommonType>(var, part[i].getParticleIDall(), adios2::Mode::Deferred);
    }
}

// restart, all of the above


};


}






#endif

