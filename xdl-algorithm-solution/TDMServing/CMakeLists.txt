cmake_minimum_required (VERSION 2.8)

project (tdm_serving)

include(${PROJECT_SOURCE_DIR}/cmake/CompileProto.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)

enable_testing()

# The default tdm-serving install prefix
set(CMAKE_INSTALL_PREFIX /opt/tdm-serving/)

# options
option(BUILD_UNIT_TESTS "BUILD_UNIT_TESTS" OFF)
option(WITH_DEBUG_SYMBOLS "WITH_DEBUG_SYMBOLS" ON)

message(STATUS "BUILD_UNIT_TESTS=${BUILD_UNIT_TESTS}")
message(STATUS "WITH_DEBUG_SYMBOLS=${WITH_DEBUG_SYMBOLS}")
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-Wall -W -fPIC -Wno-unused-but-set-variable -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG " -O0")
set(CMAKE_CXX_FLAGS_RELEASE " -O2 -DNDEBUG")

if(WITH_DEBUG_SYMBOLS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# protobuf
find_package(Protobuf REQUIRED)

# glog
find_path(GLOG_INCLUDE_PATH NAMES glog/logging.h)
find_library(GLOG_LIB NAMES glog)
if((NOT GLOG_INCLUDE_PATH) OR (NOT GLOG_LIB))
    message(FATAL_ERROR "Fail to find glog")
endif()
message("-- Found glog: " ${GLOG_LIB})

# blaze
find_path(BLAZE_INCLUDE_PATH NAMES blaze/include/predictor.h PATHS /opt)
find_library(BLAZE_LIB NAMES blaze/lib/libblaze.so PATHS /opt)
if((NOT BLAZE_INCLUDE_PATH) OR (NOT BLAZE_LIB))
    message(FATAL_ERROR "Fail to find blaze")
endif()
message("-- Found blaze: " ${BLAZE_LIB})

include_directories(
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/tdm-serving
  ${PROTOBUF_INCLUDE_DIRS}
  ${GLOG_INCLUDE_PATH}
  ${BLAZE_INCLUDE_PATH}
)

set(DYNAMIC_LIB ${PROTOBUF_LIBRARIES}
                ${GLOG_LIB}
                ${BLAZE_LIB}
)

# subdirectory
add_subdirectory(tdm-serving)
add_subdirectory(examples)
add_subdirectory(evaluation)
add_subdirectory(benchmark)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

