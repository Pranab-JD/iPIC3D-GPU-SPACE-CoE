# Install script for directory: /leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ipic3d/inputfiles" TYPE FILE FILES
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/DebyeScaleTurbulence.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/DebyeScaleTurbulenceRestart.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/Magnetosphere2D.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/MagnetotailReconEngParticle.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/NullPoints.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/Ram1TestParticle.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/ScalingTestGEM3D.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/storePersistenceDiagram.py"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/TaylorGreen.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testGEM2D_NoHDF5.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testGEM2Dsmall.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testGEM3Dsmall.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testMagnetosphere2Dsmall.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testMagnetosphere2DsmallNBCIO.inp"
    "/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/inputfiles/testMagnetosphere3Dsmall.inp"
    )
endif()

