#pragma once
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define HD __host__ __device__
#else
  #define HD
#endif

namespace moments130 
{

    // Base channels
    enum : int { CH_RHO = 0, CH_JX = 1, CH_JY = 2, CH_JZ = 3 };
    constexpr int BASE_CHANNELS = 4;

    // Exact mass-matrix stencil
    constexpr int NUM_NEIGHBORS = 14;   // matches NeNoX/Y/Z
    constexpr int MM_ROWS = 3, MM_COLS = 3;
    constexpr int MM_PER_NEIGHBOR = MM_ROWS * MM_COLS; // 9

    // Total channels per node
    constexpr int NUM_CHANNELS = BASE_CHANNELS + NUM_NEIGHBORS * MM_PER_NEIGHBOR; // 130

    // (neighbor,r,c) -> channel index
    HD inline int mm_channel(int ind, int r, int c) 
    {
        return BASE_CHANNELS + ind * MM_PER_NEIGHBOR + r * MM_COLS + c;
    }

    // 3D -> 1D node index
    HD inline std::uint64_t lin_index(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz,
                                      std::uint32_t i,  std::uint32_t j,  std::uint32_t k) 
    {
        return (static_cast<std::uint64_t>(i) * ny + j) * nz + k;
    }

    // channel base offset
    HD inline std::uint64_t chan_offset(int ch, std::uint64_t oneDensity) 
    {
        return static_cast<std::uint64_t>(ch) * oneDensity;
    }

} // namespace moments130

#undef HD

static_assert(moments130::NUM_CHANNELS == 130, "Layout must be 4 + 14*9");
