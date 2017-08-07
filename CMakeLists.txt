PROJECT(Joint-VO-SF)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# custom cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

MESSAGE(STATUS "CMAKE_MODULE_PATH:   " ${CMAKE_MODULE_PATH})

find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(SuiteSparse REQUIRED)

#Find dependencies (Eigen is included in MRPT)
find_package(MRPT REQUIRED base gui opengl obs)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(TBB REQUIRED)

set(efusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenNI2_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB ef_srcs Deformation.cpp ElasticFusion.cpp Ferns.cpp GlobalModel.cpp GPUTexture.cpp IndexMap.cpp)
file(GLOB ef_utils_srcs Utils/*.cpp)
file(GLOB ef_shader_srcs Shaders/*.cpp)
file(GLOB ef_cuda Cuda/*.cu Cuda/*.cuh)
file(GLOB ef_containers Cuda/containers/*.cpp)

set(CUDA_ARCH_BIN "20 30 35 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(ef_cuda_objs ${ef_cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11 -DSHADER_DIR=${efusion_SHADER_DIR}")


ADD_LIBRARY(vo_sf_lib
   ${ef_srcs}
   ${ef_utils_srcs}
   ${ef_shader_srcs}
   ${ef_cuda} 
   ${ef_cuda_objs} 
   ${ef_containers}
	joint_vo_sf.h
	structs_parallelization.h
	kmeans.cpp
	visualization.cpp
	solver.cpp
	segmentation_background.cpp
	camera.cpp
	camera.h
	datasets.cpp
	datasets.h
   ef.h
   GUI.h
	normal_equation.cpp
	opencv_ext.cpp
)
	
TARGET_LINK_LIBRARIES(vo_sf_lib
	${MRPT_LIBS}
	${OpenNI2_LIBRARY}
	${OpenCV_LIBS}
   ${TBB_LIBRARIES}
   ${Pangolin_LIBRARIES}
   ${CUDA_LIBRARIES}
   ${SUITESPARSE_LIBRARIES}
)
	
		
#To run online with an RGB-D camera	
ADD_EXECUTABLE(VO-SF-Camera 	main_vo_sf_camera.cpp)	
TARGET_LINK_LIBRARIES(VO-SF-Camera 	vo_sf_lib)

		
#To test it with the TUM dataset		
ADD_EXECUTABLE(VO-SF-Datasets 	main_vo_sf_datasets.cpp)
TARGET_LINK_LIBRARIES(VO-SF-Datasets 	vo_sf_lib)
		
		
#To test it with individual Image pairs		
ADD_EXECUTABLE(VO-SF-ImagePair 	main_vo_sf_imagepair.cpp)
TARGET_LINK_LIBRARIES(VO-SF-ImagePair	vo_sf_lib)
		

#To test it with pre-recorded RGB-D sequences		
ADD_EXECUTABLE(VO-SF-ImageSeq 	main_vo_sf_imageseq.cpp)
TARGET_LINK_LIBRARIES(VO-SF-ImageSeq 	vo_sf_lib)
			



# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_BUILD_TYPE "Release") #I'm not sure if this does anything
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -mavx -std=c++11")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

