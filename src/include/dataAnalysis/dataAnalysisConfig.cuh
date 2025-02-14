#ifndef _DATA_ANALYSIS_CONFIG_H_
#define _DATA_ANALYSIS_CONFIG_H_

#include <string>
#include "cudaTypeDef.cuh"

#pragma once

// General configuration
inline constexpr bool DATA_ANALYSIS_ENABLED = true;
inline constexpr bool VELOCITY_HISTOGRAM_ENABLE = true;
inline constexpr bool GMM_ENABLE = true;

inline const std::string DATA_ANALYSIS_OUTPUT_DIR = "./";
inline constexpr int DATA_ANALYSIS_EVERY_CYCLE = 100; // 0 to disable

// Histogram configuration
inline constexpr bool HISTOGRAM_OUTPUT = true;
inline const std::string HISTOGRAM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityHistogram/";

inline constexpr bool HISTOGRAM_FIXED_RANGE = true; // edit the range in velocityHistogram::getRange --> moved here
inline constexpr cudaCommonType MIN_VELOCITY_HIST_E = -0.2;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_E = 0.2;
inline constexpr cudaCommonType MIN_VELOCITY_HIST_I = -0.09;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_I = 0.09;

inline constexpr bool HISTOGRAM_OUTPUT_3D = false; // the vtk file format, if false the 3 planes are on the same surface in paraview

// GMM configuration
inline constexpr bool GMM_OUTPUT = true;
inline const std::string GMM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityGMM/";
inline constexpr int DATA_DIM = 2; // only works with DATA_DIM = 2 now
inline constexpr int NUM_COMPONENT_GMM = 8;
inline constexpr int MAX_ITERATION_GMM = 100;
inline constexpr cudaCommonType  THRESHOLD_CONVERGENCE_GMM = 1e-6;
inline constexpr bool CHECK_COVMATRIX_GMM = true;
inline constexpr bool NORMALIZE_DATA_FOR_GMM = true;


constexpr bool checkDAEnabled(){
    if constexpr(GMM_ENABLE){
        static_assert(VELOCITY_HISTOGRAM_ENABLE, "GMM requires velocity histogram to be enabled");
    }
    return true;
}

inline auto discard = checkDAEnabled();



#endif