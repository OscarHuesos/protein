cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

#CXXFLAGS = -O -std=c++11
#LDFLAGS = -O -std=c++11
#set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard")
project(protein)

###############################################################################
#option(ROOT "Include ROOT" ON)
#################################################################################
option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB pdb_files "${CMAKE_SOURCE_DIR}/bin/pdbs/*.pdb")
#file(GLOB txt_files "${CMAKE_SOURCE_DIR}/pdbs/*.txt")

#file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/pdbs/*.pdb")
#file(COPY ${CMAKE_SOURCE_DIR}/pdbs
 #    DESTINATION ${CMAKE_BINARY_DIR})

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/bin
    )
set(
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/lib
    )

if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
    if(USE_CUDA)
        message(mODERN CUDA)
        enable_language("CUDA")
        add_executable(protein ${cpu_source_files} ${gpu_source_files} ${pdb_files} ${txt_files})
	#add_executable(protein ${cpu_source_files} ${gpu_source_files})
    else(USE_CUDA)
        add_executable(protein ${cpu_source_files}  ${pdb_files} ${txt_files})
    endif()
else()
    # Old CMake
    add_executable(protein ${cpu_source_files}  ${pdb_files} ${txt_files})
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            CUDA_ADD_LIBRARY(protein_gpu ${gpu_source_files})
            target_link_libraries(protein protein_gpu)
        endif()
    endif()
endif()
###########################################################################
#if (ROOT)
 # message(STATUS "USE_ROOT: ${ROOT}")
 # set(DEFINITIONS "${DEFINITIONS} -DROOT")
  #add_definitions(-DROOT)
#endif()

#if (ROOT)
# message(mODERN CUDA USA ROOT)
# You need to tell CMake where to find the ROOT installation. This can be done in a number of ways:
#   - ROOT built with classic configure/make use the provided $ROOTSYS/etc/cmake/FindROOT.cmake
#   - ROOT built with CMake. Add in CMAKE_PREFIX_PATH the installation prefix for ROOT
# list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
# find_package(ROOT REQUIRED)
# set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
 #---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
#---setup ROOT include + lib dirs
#  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIRS})
#  link_directories(${ROOT_LIBRARY_DIR})
#  set(RNGTEST_LIBRARIES_EXTERNAL ${RNGTEST_LIBRARIES_EXTERNAL} ${ROOT_LIBRARIES})
 # set(RNGTEST_EXTERNAL_INCLUDES "${RNGTEST_EXTERNAL_INCLUDES};${ROOT_INCLUDE_DIRS}")
#endif()
################################################################################

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
