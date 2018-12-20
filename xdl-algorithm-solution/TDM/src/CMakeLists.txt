cmake_minimum_required(VERSION 3.3.0)
project(dist_tree)

include(FindPythonAnaconda.cmake)

add_definitions(-std=c++1y -D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "-O3 -g -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

enable_testing()

execute_process(COMMAND python -c "import xdl; print(xdl.include_path)" OUTPUT_VARIABLE XDL_INCLUDE_PATH)
execute_process(COMMAND python -c "import xdl; print(xdl.lib_path)" OUTPUT_VARIABLE XDL_LIB_PATH)
execute_process(COMMAND python -c "import xdl; print(xdl.bin_path)" OUTPUT_VARIABLE XDL_BIN_PATH)

string(REGEX REPLACE "\n$" "" XDL_INCLUDE_PATH  "${XDL_INCLUDE_PATH}")
string(REGEX REPLACE "\n$" "" XDL_LIB_PATH  "${XDL_LIB_PATH}")
string(REGEX REPLACE "\n$" "" XDL_BIN_PATH  "${XDL_BIN_PATH}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()
find_package (Threads)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(googletest)
include(glog)
include(benchmark)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${XDL_INCLUDE_PATH}
                    ${GTEST_INCLUDE_DIRS}
		            ${GLOG_INCLUDE_DIRS}
		            ${TBSYS_INCLUDE_DIRS}
		            ${BENCHMARK_INCLUDE_DIRS})
find_program(
  PROTOC_BIN
  NAME protoc
  PATHS ${XDL_BIN_PATH}
  NO_DEFAULT_PATH
)

message(STATUS "Finding xdl lib ...")
find_library(
  XDL_LIB
  NAMES xdl_python_pybind
  PATHS ${XDL_LIB_PATH})
message(STATUS "xdl lib: ${XDL_LIB}")

add_subdirectory(tdm)
add_subdirectory(python)
