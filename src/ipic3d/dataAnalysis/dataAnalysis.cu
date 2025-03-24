
#include <thread>
#include <future>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "iPic3D.h"
#include "VCtopology3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "GMM/cudaGMMUtility.cuh"
#include "GMM/cudaGMM.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"

#include "mpi.h"

namespace dataAnalysis
{

using namespace iPic3D;
using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 3>;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;

using namespace DAConfig;

class dataAnalysisPipelineImpl {
private:
    int ns;
    int deviceOnNode;
    // pointers to objects in KCode
    cudaStream_t* streams;
    particleArrayCUDA** pclsArrayHostPtr = nullptr;

    std::future<int> analysisFuture;

    ThreadPool* DAthreadPool = nullptr;
    velocitySoA* velocitySoACUDA = nullptr;
    // histogram
    string HistogramSubDomainOutputPath;
    velocityHistogram::velocityHistogram* velocityHistogram = nullptr;

    // GMM
    string GMMSubDomainOutputPath;
    cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>* gmmArray = nullptr;
    // dimensions: numSpecies, numPlanes (uvw), cycles
    std::vector<std::array<std::vector<cudaGMMWeight::GMMResult<GMMType, DATA_DIM_GMM>>, 3>> gmmResults;

public:

    dataAnalysisPipelineImpl(c_Solver& KCode) {
        ns = KCode.ns;
        deviceOnNode = KCode.cudaDeviceOnNode;
        streams = KCode.streams;
        pclsArrayHostPtr = KCode.pclsArrayHostPtr;

        DAthreadPool = new ThreadPool(4);

        if constexpr (VELOCITY_HISTOGRAM_ENABLE) { // velocity histogram
            velocitySoACUDA = new velocitySoA();

            HistogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "_";
            velocityHistogram = new velocityHistogram::velocityHistogram(VELOCITY_HISTOGRAM_RES*VELOCITY_HISTOGRAM_RES);

            if constexpr (GMM_ENABLE) { // GMM
                GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "_";
                gmmArray = new cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>[3];
                gmmResults.resize(ns);
            }
        }
    }

    void startAnalysis(int cycle);

    int checkAnalysis();

    int waitForAnalysis();


    ~dataAnalysisPipelineImpl() {     

        if (DAthreadPool != nullptr) delete DAthreadPool;
        if (velocitySoACUDA != nullptr) delete velocitySoACUDA;
        if (velocityHistogram != nullptr) delete velocityHistogram;
        if (gmmArray != nullptr) delete[] gmmArray;
    }

private:

    int analysisEntre(int cycle);

    int GMMAnalysisSpecies(const int cycle, const int species, const std::string outputPath);
    
};



/**
 * @brief analysis function for each species, uv, uw, vw
 * @details It launches 3 threads for uv uw vw analysis in parallel
 * 
 */
int dataAnalysisPipelineImpl::GMMAnalysisSpecies(const int cycle, const int species, const std::string outputPath){

    using weightType = cudaTypeSingle;

    std::future<int> future[3];

    auto GMMLambda = [=](int i) mutable {

        using namespace cudaGMMWeight;

        cudaErrChk(cudaSetDevice(deviceOnNode));

        // GMM config

        // set the random number generator to sample velocity from circle of radius max velocity
        std::random_device rd;  // True random seed
        std::mt19937 gen(rd()); // Mersenne Twister PRN
        std::uniform_real_distribution<GMMType>  unif01(0.0, 1.0);
        std::uniform_real_distribution<GMMType> distTheta(0, 2*M_PI);

        const GMMType maxVelocity = (species == 0 || species == 2) ? MAX_VELOCITY_HIST_E : MAX_VELOCITY_HIST_I;
        // it is assumed that DATA_DIM_GMM == 2 and that the velocity range is homogenues in all dimensions
        const GMMType maxVelocityArray[DATA_DIM_GMM] = {maxVelocity,maxVelocity};
        // right now it is not used (fixed to zero) --> mean is not subtracted to data, but it might be useful for future developments
        const GMMType meanArray[DATA_DIM_GMM] = {0.0,0.0};

        const GMMType uth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType vth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType wth = species == 0 || species == 2 ? 0.045 : 0.0126;
        
        GMMType var1 = 0.01;
        GMMType var2 = 0.01; 

        GMMType weightVector[NUM_COMPONENT_GMM];
        GMMType meanVector[NUM_COMPONENT_GMM * DATA_DIM_GMM];
        GMMType coVarianceMatrix[NUM_COMPONENT_GMM * DATA_DIM_GMM * DATA_DIM_GMM ];
        
        if (i==0){
            var1 = uth*uth; 
            var2 = vth*vth;
        }
        else if(i==1){
            var1 = vth*vth;
            var2 = wth*wth;
        }
        else if(i==2){
            var1 = uth*uth;
            var2 = wth*wth;
        }
        
        // normalize initial parameters if NORMALIZE_DATA_FOR_GMM==true
        GMMType normalization = 1.0; 
        if constexpr(NORMALIZE_DATA_FOR_GMM) normalization = maxVelocity;

        if constexpr (START_WITH_LAST_PARAMETERS_GMM) // start GMM with output GMM parameters from last cycle as initial parameters
        {
            // if first time initialize GMM with the usual fixed parameters
            if(gmmResults[species][i].size() == 0)
            {                
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    GMMType radius = maxVelocity * sqrt(unif01(gen));
                    GMMType theta = distTheta(gen);
                    meanVector[j * 2] =  radius*cos(theta);
                    meanVector[j * 2 + 1] = radius*sin(theta);
                    coVarianceMatrix[j * 4] = var1;
                    coVarianceMatrix[j * 4 + 1] = 0.0;
                    coVarianceMatrix[j * 4 + 2] = 0.0;
                    coVarianceMatrix[j * 4 + 3] = var2;
                }
            }
            else // Initialize GMM with previous output parameters
            {   
                auto& lastResult = gmmResults[species][i].back();

                bool reset = false;
                
                // safety checks on components weights and mean since after pruning some components have zero weight
                // check if meanVector is NaN or component weight is too small
                // if meanVector is NaN sample new mean vector
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){ 
                    if( std::isnan(lastResult.mean[j * 2]) || std::isnan(lastResult.mean[j * 2 + 1]) || 
                        std::isinf(lastResult.mean[j * 2]) || std::isinf(lastResult.mean[j * 2 + 1]) ){
                        reset = true;
                        GMMType radius = maxVelocity * sqrt(unif01(gen));
                        GMMType theta = distTheta(gen);
                        meanVector[j * 2] =  radius*cos(theta) / normalization;
                        meanVector[j * 2 + 1] = radius*sin(theta) / normalization;
                    }
                    else{
                        meanVector[j * 2] = lastResult.mean[j * 2] / normalization;
                        meanVector[j * 2 + 1] = lastResult.mean[j * 2 + 1] / normalization;
                    }

                    if( lastResult.weight[j] < PRUNE_THRESHOLD_GMM*10 ) reset = true;
                }

                // if meanVector is NaN or weigth is too small reset components weights
                // adjust cov if it is too small
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    if(reset){
                        weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    }
                    else{
                        weightVector[j] = lastResult.weight[j];
                    }

                    coVarianceMatrix[j * 4] = lastResult.coVariance[j * 4] > (10 * EPS_COVMATRIX_GMM * (normalization*normalization)) ? 
                                                lastResult.coVariance[j * 4] : var1;
                    coVarianceMatrix[j * 4] /= (normalization*normalization);                               
                    coVarianceMatrix[j * 4 + 1] = 0.0;
                    coVarianceMatrix[j * 4 + 2] = 0.0;
                    coVarianceMatrix[j * 4 + 3] = lastResult.coVariance[j * 4 + 3] > (10 * EPS_COVMATRIX_GMM * (normalization*normalization)) ? 
                                                    lastResult.coVariance[j * 4 + 3] : var2;
                    coVarianceMatrix[j * 4 + 3] /= (normalization*normalization);
                }
            }
        }
        else // start GMM with fixed initial parameters at any cycle
        {
            for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                GMMType radius = maxVelocity * sqrt(unif01(gen));
                GMMType theta = distTheta(gen);
                meanVector[j * 2] =  radius*cos(theta) / normalization;
                meanVector[j * 2 + 1] = radius*sin(theta) / normalization;
                coVarianceMatrix[j * 4] = var1 / (normalization*normalization);
                coVarianceMatrix[j * 4 + 1] = 0.0;
                coVarianceMatrix[j * 4 + 2] = 0.0;
                coVarianceMatrix[j * 4 + 3] = var2 / (normalization*normalization);
            }
        }

        GMMParam_t<GMMType> GMMParam = {
            .numComponents = NUM_COMPONENT_GMM,
            .maxIteration = MAX_ITERATION_GMM,
            .threshold = THRESHOLD_CONVERGENCE_GMM,
            .weightInit = weightVector,
            .meanInit = meanVector,
            .coVarianceInit = coVarianceMatrix
        };  


        // data
        GMMDataMultiDim<GMMType, DATA_DIM_GMM, weightType> GMMData
            (VELOCITY_HISTOGRAM_RES*VELOCITY_HISTOGRAM_RES, velocityHistogram->getHistogramScaleMark(i), velocityHistogram->getVelocityHistogramCUDAArray(i), 
            {maxVelocityArray[0], maxVelocityArray[1]});

        cudaErrChk(cudaHostRegister(&GMMData, sizeof(GMMData), cudaHostRegisterDefault));
        
        // generate exact output file path        
        auto& gmm = gmmArray[i];
        gmm.config(&GMMParam, &GMMData);
        // preprocess the data --> normalize data
        gmm.preProcessDataGMM(meanArray);
        // run GMM
        auto convergStep = gmm.initGMM(); // the exact output file name
        // postporcess data --> normalize data back
        gmm.postProcessDataGMM();
        // move GMM output to host
        gmmResults[species][i].push_back(gmm.getGMMResult(cycle, convergStep));

        cudaErrChk(cudaHostUnregister(&GMMData));

        return 0;
    };

    for(int i = 0; i < 3; i++){
        future[i] = DAthreadPool->enqueue(GMMLambda, i); 
    }

    for(int i = 0; i < 3; i++){
        future[i].wait();
    }

    // GMM result output, in the gmmResult
    if constexpr (GMM_OUTPUT) {
        auto& uvw = DA_2D_PLANE_NAME;

        if (!std::filesystem::exists(outputPath)){ 
            std::ofstream output(outputPath);
            if(output.is_open()){
                output.close();
            } else {
                throw std::runtime_error("[!]Error: Can not open output file for velocity GMM species");
            }
        }

        std::fstream output(outputPath, std::ios::in | std::ios::out | std::ios::ate);
        if(output.is_open()){
            std::ostringstream buffer;

            if(output.tellg() == 0) buffer << "{\n"; 
            else { 
                output.seekp(-2, std::ios_base::end);
                buffer << ",\n";
            }

            buffer << "\"" << std::to_string(cycle) << "\": {\n";
            buffer << "\"" << uvw[0] << "\": " << gmmResults[species][0].back().outputString() << ",\n";
            buffer << "\"" << uvw[1] << "\": " << gmmResults[species][1].back().outputString() << ",\n";
            buffer << "\"" << uvw[2] << "\": " << gmmResults[species][2].back().outputString() << "\n";
            buffer << "}\n}";
            
            output << buffer.str();
            output.close();
        } else {
            throw std::runtime_error("[!]Error: Can not open output file for velocity GMM species");
        }

    }

    return 0;
}

/**
 * @brief analysis function, called by startAnalysis
 * @details procesures in this function should be executed in sequence, the order of the analysis should be defined here
 *          But the procedures can launch other threads to do the analysis
 *          Also this function is a friend function of c_Solver, resources in the c_Slover should be dispatched here
 */
int dataAnalysisPipelineImpl::analysisEntre(int cycle){
    cudaErrChk(cudaSetDevice(deviceOnNode));

    // species by species to save VRAM
    for(int i = 0; i < ns; i++){
        if constexpr (VELOCITY_HISTOGRAM_ENABLE) {
            // to SoA
            velocitySoACUDA->updateFromAoS(pclsArrayHostPtr[i], streams[i]);

            // histogram
            auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "_";
            velocityHistogram->init(velocitySoACUDA, cycle, i, streams[i]);
            if constexpr (HISTOGRAM_OUTPUT)
            velocityHistogram->writeToFile(histogramSpeciesOutputPath, streams[i]); 
            else cudaErrChk(cudaStreamSynchronize(streams[i]));

            if constexpr (GMM_ENABLE) { // GMM
                auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + ".json";
                GMMAnalysisSpecies(cycle, i, GMMSpeciesOutputPath);
            }
        }
    }

    
    return 0;
}


/**
 * @brief start all the analysis registered here
 */
void dataAnalysisPipelineImpl::startAnalysis(int cycle){

    if(DATA_ANALYSIS_EVERY_CYCLE == 0 || (cycle % DATA_ANALYSIS_EVERY_CYCLE != 0)){
        analysisFuture = std::future<int>();
    } else {
        analysisFuture = DAthreadPool->enqueue(&dataAnalysisPipelineImpl::analysisEntre, this, cycle); 

        if(analysisFuture.valid() == false){
            throw std::runtime_error("[!]Error: Can not start data analysis");
        }
    }

}

/**
 * @brief check if the analysis is done, non-blocking
 * 
 * @return 0 if the analysis is done, 1 if it is not done
 */
int dataAnalysisPipelineImpl::checkAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    if(analysisFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
        return 0;
    }else{
        return 1;
    }

    return 0;
}

/**
 * @brief wait for the analysis to be done, blocking
 */
int dataAnalysisPipelineImpl::waitForAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    analysisFuture.wait();

    return 0;
}


/**
 * @brief create output directory for the data analysis, controlled by dataAnalysisConfig.cuh
 */
void dataAnalysisPipeline::createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct){ // output path for data analysis
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }

    // VCT mapping for this subdomain
    auto writeVctMapping = [&](const std::string& filePath) {
        std::ofstream vctMapping(filePath);
        if(vctMapping.is_open()){
        vctMapping << "Cartesian rank: " << vct->getCartesian_rank() << std::endl;
        vctMapping << "Number of processes: " << vct->getNprocs() << std::endl;
        vctMapping << "XLEN: " << vct->getXLEN() << std::endl;
        vctMapping << "YLEN: " << vct->getYLEN() << std::endl;
        vctMapping << "ZLEN: " << vct->getZLEN() << std::endl;
        vctMapping << "X: " << vct->getCoordinates(0) << std::endl;
        vctMapping << "Y: " << vct->getCoordinates(1) << std::endl;
        vctMapping << "Z: " << vct->getCoordinates(2) << std::endl;
        vctMapping << "PERIODICX: " << vct->getPERIODICX() << std::endl;
        vctMapping << "PERIODICY: " << vct->getPERIODICY() << std::endl;
        vctMapping << "PERIODICZ: " << vct->getPERIODICZ() << std::endl;

        vctMapping << "Neighbor X left: " << vct->getXleft_neighbor() << std::endl;
        vctMapping << "Neighbor X right: " << vct->getXright_neighbor() << std::endl;
        vctMapping << "Neighbor Y left: " << vct->getYleft_neighbor() << std::endl;
        vctMapping << "Neighbor Y right: " << vct->getYright_neighbor() << std::endl;
        vctMapping << "Neighbor Z left: " << vct->getZleft_neighbor() << std::endl;
        vctMapping << "Neighbor Z right: " << vct->getZright_neighbor() << std::endl;

        vctMapping.close();
        } else {
        throw std::runtime_error("[!]Error: Can not create VCT mapping for velocity GMM species");
        }
    };

    if constexpr (VELOCITY_HISTOGRAM_ENABLE && HISTOGRAM_OUTPUT) {
        const auto histogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR;

        if(myrank == 0 && 0 != checkOutputFolder(histogramSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity histogram");
        }

        MPI_Barrier(MPIdata::get_PicGlobalComm());

        writeVctMapping(histogramSubDomainOutputPath + "vctMapping_subDomain" + std::to_string(myrank) + ".txt");
    }

    if constexpr (GMM_ENABLE && GMM_OUTPUT) {
        const auto GMMSubDomainOutputPath = GMM_OUTPUT_DIR;

        if(myrank == 0 && 0 != checkOutputFolder(GMMSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity GMM");
        }

        MPI_Barrier(MPIdata::get_PicGlobalComm());

        writeVctMapping(GMMSubDomainOutputPath + "vctMapping_subDomain" + std::to_string(myrank) + ".txt");
    }

}



dataAnalysisPipeline::dataAnalysisPipeline(iPic3D::c_Solver& KCode) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        impl = nullptr;
        return;
    }
    impl = std::make_unique<dataAnalysisPipelineImpl>(std::ref(KCode));
}

void dataAnalysisPipeline::startAnalysis(int cycle) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }
    impl->startAnalysis(cycle);
}

int dataAnalysisPipeline::checkAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->checkAnalysis();
}

int dataAnalysisPipeline::waitForAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->waitForAnalysis();
}


dataAnalysisPipeline::~dataAnalysisPipeline() {
    
}
    
} // namespace dataAnalysis







