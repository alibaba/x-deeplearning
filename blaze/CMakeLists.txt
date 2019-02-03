project(blaze)

cmake_minimum_required(VERSION 2.8)

option(test "Build all tests." ON)
enable_testing()
set(CMAKE_VERBOSE_MAKEFILE ON)

include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
# The default blaze install prefix
set(CMAKE_INSTALL_PREFIX /opt/blaze/)

######################################################
blaze_option(USE_CUDA "USE_CUDA" ON)
blaze_option(USE_MKL "USE_MKL" ON)
blaze_option(SYMBOL_EXPORT_CTL "Symbol Export Control" OFF)
blaze_option(COMPILE_STATIC "Compile static library" OFF)
blaze_option(COVERAGE "Enable converage" OFF)

message(STATUS "Build summary:")
message(STATUS "USE_CUDA=${USE_CUDA}")
message(STATUS "USE_MKL=${USE_MKL}")
message(STATUS "SYMBOL_EXPORT_CTL=${SYMBOL_EXPORT_CTL}")
message(STATUS "COMPILE_STATIC=${COMPILE_STATIC}")
message(STATUS "COVERAGE=${COVERAGE}")
#######################################################
if (USE_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DUSE_CUDA")

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
find_package(CUDA REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/Cuda.cmake)
include_directories(
    ${CUDA_PATH}/include/
    ${CUDNN_ROOT_DIR}/include
)
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/
    ${CUDNN_ROOT_DIR}/lib64/
)
endif()

if (USE_MKL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")

set(MKL_TOOLKIT_ROOT_DIR ${MKL_PATH})
if (EXISTS "${MKL_TOOLKIT_ROOT_DIR}/mkl")
  include_directories(
    ${MKL_TOOLKIT_ROOT_DIR}/mkl/include
  )
  link_directories(
    ${MKL_TOOLKIT_ROOT_DIR}/mkl/lib/intel64
    ${MKL_TOOLKIT_ROOT_DIR}/lib/intel64
  )
else()
  include_directories(
    ${MKL_TOOLKIT_ROOT_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/mkl_dnn/include/
  )
  link_directories(${MKL_TOOLKIT_ROOT_DIR}/lib64)
endif()
else()
  include_directories(
    ${OPENBLAS_PATH}/include/openblas
    )
  link_directories(
    ${OPENBLAS_PATH}/lib
    ${OPENBLAS_PATH}/lib64
    )
endif()

set(PROTOBUF_VERSION "protobuf-3.6.0")
set(ONNX_VERSION "onnx-1.2.2")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -DNDEBUG -Xcompiler -fPIC -D_FORCE_INLINES --expt-extended-lambda --ptxas-options=-v")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG -std=c++11 -fPIC -ftree-vectorize -Wcpp -mavx -mavx2 -mf16c -fopenmp -g -DHAVE_PTHREAD")

include_directories(
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_BINARY_DIR}/blaze/model_importer/
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/blaze/
  ${PROJECT_SOURCE_DIR}/thirdparty/
  ${PROJECT_SOURCE_DIR}/gtest/
  ${PROJECT_SOURCE_DIR}/thirdparty/protobuf/${PROTOBUF_VERSION}/src/
  ${PROJECT_SOURCE_DIR}/thirdparty/zlib/zlib-1.2.8/
  ${PROJECT_SOURCE_DIR}/thirdparty/onnx/1.2.2/include/  
)

# set protobuf
set(PROTOBUF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/protobuf/${PROTOBUF_VERSION}/src/)
set(PROTOBUF_PROTOC_EXECUTABLE ${PROJECT_SOURCE_DIR}/thirdparty/protobuf/${PROTOBUF_VERSION}/bin/protoc)
set(PROTOBUF_LIBRARIES Tricky)
set(PROTOBUF_IMPORT_DIRS
  ${PROJECT_SOURCE_DIR}/thirdparty/protobuf/${PROTOBUF_VERSION}/src/
)
find_package(Protobuf REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/ProtoBuf.cmake)

if (COVERAGE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

add_subdirectory(thirdparty)
add_subdirectory(blaze)
add_subdirectory(tools)
add_subdirectory(example)

if (SYMBOL_EXPORT_CTL)
else()
  add_subdirectory(serving)
endif()
