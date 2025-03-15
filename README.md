# iPIC3D

The Particle Pusher and Moment Gatherer runs on the GPU whilst the Fied Solver runs on the CPU.

## Requirements
  - g++/nvcc compiler (to support C++ 17)
  - cmake (minimum version 2.8)
  - MPI (OpenMPI or MPICH)
  - CUDA (7.5 or higher) or HIP
  - HDF5 (optional)
  - Paraview/Catalyst (optional)

## Installation
1. Download the code
``` shell
git clone https://github.com/Pranab-JD/iPIC3D-GPU-SPACE-CoE.git
```

2. Create build directory
``` shell
cd iPIC3D-GPU-SPACE-CoE && mkdir build && cd build
```

3. Compile the code using CUDA
``` shell
cmake ..    # CUDA is enabled by default
make -j     # -j = build with max # of threads - fast, recommended
```

or compile the code using HIP
``` shell
cmake -DHIP_ON=ON ..    # use HIP
make -j                 # -j = build with max # of threads - fast, recommended
```

If you use HIP, change the GPU architecture in [CMakeLists.txt](./CMakeLists.txt) according to your hardware to get best performance
``` cmake 
set_property(TARGET iPIC3Dlib PROPERTY HIP_ARCHITECTURES gfx90a) 
```

4. Run
``` shell
# no_of_proc = XLEN x YLEN x ZLEN (as specified in the input file)
mpirun -np no_of_proc ./iPIC3D  inputfilename.inp
```

**Important:** make sure `number of MPI process = XLEN x YLEN x ZLEN` as specified in the input file.

On a multi-node supercomputer, you may have to use `srun` to launch the program. 

### OpenMP
As the field solver runs on CPU, its performance can be enhanced using OpenMP. Set the desired number of OpenMP threads per MPI task.
``` shell
export OMP_NUM_THREADS=4    # set OpenMP threads for each MPI process
```

# Citation
Markidis, Stefano and Giovanni Lapenta (2010), *Multi-scale simulations of plasma with iPIC3D*, Mathematics and Computers in Simulation, 80, 7, 1509-1519 [[DOI]](https://doi.org/10.1016/j.matcom.2009.08.038)