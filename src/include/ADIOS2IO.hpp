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

using optionFuncType = std::function<void(adios2::IO&, adios2::Engine&)>;

class ADIOS2Manager {

private:
    // adios2
    adios2::ADIOS adios;

    adios2::IO ioField;
    adios2::Engine engineField;
    std::vector <optionFuncType> fieldOptions;

    adios2::IO ioParticle;
    adios2::Engine engineParticle;
    std::vector <optionFuncType> particleOptions;
    
    adios2::IO ioRestart;
    adios2::Engine engineRestart;
    std::vector <optionFuncType> restartOptions;

    std::unordered_map<std::string, optionFuncType> outputTagOptions;

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

    ADIOS2Manager() {
        outputTagOptions = {
            {"position", std::bind(&ADIOS2Manager::_particlePosition, this, std::placeholders::_1, std::placeholders::_2)},
            {"velocity", std::bind(&ADIOS2Manager::_particleVelocity, this, std::placeholders::_1, std::placeholders::_2)},
            {"q", std::bind(&ADIOS2Manager::_particleCharge, this, std::placeholders::_1, std::placeholders::_2)},
            {"ID", std::bind(&ADIOS2Manager::_particleID, this, std::placeholders::_1, std::placeholders::_2)}
        };
    }

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
        
    }

    return var;            
}

// tag mapping
/* Field
    collective
    total_topology 
    proc_topology
    B --> to write all B components
    E --> to write all E components
    phi --> scalar vector
    Jall --> to write all J (current density) components
    Jsall --> to write all Js (current densities for each species) components
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
void _particlePosition(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto x = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionX", {sizeNOP}, {0}, {sizeNOP});
        auto y = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionY", {sizeNOP}, {0}, {sizeNOP});
        auto z = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionZ", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(x, part[i].getXall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(y, part[i].getYall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(z, part[i].getZall(), adios2::Mode::Deferred);
    }
}

void _particleVelocity(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto u = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityU", {sizeNOP}, {0}, {sizeNOP});
        auto v = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityV", {sizeNOP}, {0}, {sizeNOP});
        auto w = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityW", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(u, part[i].getUall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(v, part[i].getVall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(w, part[i].getWall(), adios2::Mode::Deferred);
    }
}

void _particleCharge(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto var = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "charge", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(var, part[i].getQall(), adios2::Mode::Deferred);
    }
}

void _particleID(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i].getNOP());

        auto var = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "ID", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(var, part[i].getParticleIDall(), adios2::Mode::Deferred);
    }
}

// restart, all of the above


};


}






#endif

