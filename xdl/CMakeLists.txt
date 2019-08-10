project(xdl)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)

xdl_option(test      "Build all tests."          ON)
xdl_option(USE_GPU "use gpu" off)
xdl_option(USE_PS_PLUS "use ps-plus" ON)
xdl_option(coverage  "Generate coverage analysis"  off)

enable_testing()
set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

IF (USE_GPU)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -std=c++1y -Wcpp -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_USE_THREADS -DEIGEN_USE_GPU -DUSE_GPU")
ELSE ()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -std=c++1y -Wcpp -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_USE_THREADS")
ENDIF ()

IF (coverage)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(OPT_FLAGS "-g3")
    set(GPU_OPT_FLAGS "")
ELSE ()
    set(OPT_FLAGS "-O2 -g")
    set(GPU_OPT_FLAGS "-O3")
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "build on debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb -pipe -fopenmp")
    set(XDL_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin ${NVCC_C_COMPILER} -Xcompiler;-fPIC; -g -O0 -std=c++11;-D_GLIBCXX_USE_CXX11_ABI=0;-w;")
ELSE ()
    message(STATUS "build on release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS} -pipe -fopenmp -DNDEBUG")
    set(XDL_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${GPU_OPT_FLAGS} -ccbin ${NVCC_C_COMPILER} -Xcompiler;-fPIC;-std=c++11;-D_GLIBCXX_USE_CXX11_ABI=0;-w;")
ENDIF()

## manually set install path
set(install_path ${PROJECT_SOURCE_DIR}/deploy/)

include_directories(
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_BINARY_DIR}/
)

include(cmake/Dependencies.cmake)

set(TBB_ROOT "${PROJECT_SOURCE_DIR}/third_party/tbb/")
include(${TBB_ROOT}/cmake/TBBBuild.cmake)
set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS)
find_package(TBB REQUIRED)
include_directories("${TBB_ROOT}/include")

IF (USE_GPU)
   set(CUDA_PATH "/usr/local/cuda-9.0/")
   set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
   find_package(CUDA REQUIRED)
   message("-- CUDA_PATH = ${CUDA_PATH} ")
   include(${PROJECT_SOURCE_DIR}/cmake/Cuda.cmake)
   include_directories(
     ${CUDA_PATH}/targets/x86_64-linux/include/
     ${CUDA_PATH}/include/
     )
   link_directories(
     ${CUDA_PATH}/lib64/
     ${CUDA_PATH}/lib64/stubs/
     )
ENDIF ()

# PS_PLUS
set(SEASTAR_LIBRARYS -Wl,--whole-archive seastar_service ps_network_static seastar -Wl,--no-whole-archive -L/usr/local/lib64/boost -lboost_timer -lboost_chrono  -laio -lboost_program_options -lboost_system -lboost_filesystem -lm -lboost_thread -lcryptopp -lrt -lgnutls -lgnutlsxx -llz4 -ldl -lgcc_s -lunwind -lhwloc -lnuma -lpciaccess -lxml2 -lz -lcares-seastar libstdc++.a)
set(PS_LIBRARYS -Wl,--whole-archive libzookeeper.a libhashtable.a ps_common ps_client ps_server ps_scheduler ps_model_server ps_plugin_hdfs libevent_core.a -Wl,--no-whole-archive ${SEASTAR_LIBRARYS})
include_directories(${PROJECT_SOURCE_DIR}/ps-plus/)
include_directories(${PROJECT_SOURCE_DIR}/third_party/zookeeper-client/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/zookeeper-client/generated)
link_directories(${PROJECT_BINARY_DIR}/ps-plus/)
link_directories(${PROJECT_BINARY_DIR}/third_party/zookeeper-client/)
add_subdirectory(ps-plus)

# backend
xdl_option(TF_BACKEND "use tensorflow as backend" off)
xdl_option(MXNET_BACKEND "use mxnet as backend" off)

set(BACKEND_LIB "")
set(TF_BACKEND_IMPORT "")
set(MXNET_BACKEND_IMPORT "")
IF (TF_BACKEND)
   execute_process(COMMAND python -c "from __future__ import print_function;import tensorflow as tf; print(tf.sysconfig.get_include(),end='')" OUTPUT_VARIABLE TF_INCLUDE)
   set(TF_INCLUDE_DIR ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
   execute_process(COMMAND python -c "from __future__ import print_function;import tensorflow as tf; print(tf.sysconfig.get_lib(),end='')" OUTPUT_VARIABLE TF_LIB_DIR)
   set(TF_LIB -Wl,--no-as-needed tensorflow_framework ${TF_LIB_DIR}/python/_pywrap_tensorflow_internal.so -Wl,--as-needed)
   list(APPEND BACKEND_LIB ${TF_LIB})
   set(TF_BACKEND_IMPORT "from xdl.python.backend.tf.tf_backend import tf_wrapper")
ENDIF()

IF (MXNET_BACKEND)
   set(MX_LIB mxnet)
   execute_process(COMMAND  python -c "from __future__ import print_function;import mxnet,os; print(os.path.dirname(mxnet.__file__),end='')" OUTPUT_VARIABLE MX_LIB_DIR)
   set(MXNET_INCLUDE ${MX_LIB_DIR})
   set(MX_INCLUDE_DIR ${MXNET_INCLUDE}/cpp-package/include ${MXNET_INCLUDE}/include ${MXNET_INCLUDE}/3rdparty/tvm/nnvm/include ${MXNET_INCLUDE}/3rdparty/tvm/dmlc-core/include)
   list(APPEND BACKEND_LIB ${MX_LIB})
   set(MXNET_BACKEND_IMPORT "from xdl.python.backend.mxnet.mxnet_backend import mxnet_wrapper")
ENDIF()

IF (USE_GPU)
   set(XDL_CORE_DEPEND_LIB libprotobuf ${PS_LIBRARYS} ${BACKEND_LIB} python2.7 cudart dl ${TBB_IMPORTED_TARGETS})
ELSE ()
   set(XDL_CORE_DEPEND_LIB libprotobuf ${PS_LIBRARYS} ${BACKEND_LIB} python2.7 dl ${TBB_IMPORTED_TARGETS})
ENDIF ()

set(XDL_IO_DEPEND_LIB dl rdkafka++ jsoncpp)
set(XDL_CORE_LIB -Wl,--whole-archive xdl_core -Wl,--no-whole-archive ${XDL_CORE_DEPEND_LIB})
set(XDL_IO_LIB -Wl,--whole-archive xdl_io -Wl,--no-whole-archive)

# python lib
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(xdl)
add_subdirectory(test)
