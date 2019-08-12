xdl_option(BUILD_SHARED "build shared library" OFF)

aux_source_directory(lib SRC_XDL_CORE_LIB)
aux_source_directory(ops/ps_ops SRC_XDL_CORE_PS_OPS)
aux_source_directory(ops/ps_ops/model_server SRC_XDL_CORE_PS_MODEL_SERVER_OPS)
aux_source_directory(framework SRC_XDL_CORE_FRAMEWORK)
aux_source_directory(framework/gpu SRC_XDL_CORE_FRAMEWORK_GPU)
aux_source_directory(grappler SRC_XDL_CORE_GRAPPLER)
aux_source_directory(utils SRC_XDL_CORE_UTILS)
aux_source_directory(lib/base SRC_XDL_BASE_LIB)
file(GLOB SRC_XDL_CORE_OPS "ops/*.cc")

set(SOURCE_LIST ${SRC_XDL_CORE_LIB} ${SRC_XDL_CORE_FRAMEWORK} ${SRC_XDL_CORE_GRAPPLER} ${SRC_XDL_CORE_OPS} ${SRC_XDL_CORE_UTILS})

IF (USE_GPU)
   file(GLOB CUDA_SOURCE "ops/*.cu")
   set(CUDA_NVCC_FLAGS ${XDL_CUDA_NVCC_FLAGS})
   #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-std=c++11;-D_GLIBCXX_USE_CXX11_ABI=0;-w")
   cuda_compile(cuda_objs ${CUDA_SOURCE})
   list(APPEND SOURCE_LIST ${cuda_objs})
   list(APPEND SOURCE_LIST ${SRC_XDL_CORE_FRAMEWORK_GPU})
ENDIF ()

aux_source_directory(backend/tf SRC_XDL_TF_BACKEND)
aux_source_directory(backend/mxnet SRC_XDL_MX_BACKEND)

# backend
IF (TF_BACKEND)
   include_directories(${TF_INCLUDE_DIR})
   link_directories(${TF_LIB_DIR})
   list(APPEND SOURCE_LIST ${SRC_XDL_TF_BACKEND})
ENDIF()

IF (MXNET_BACKEND)
   include_directories(${MX_INCLUDE_DIR})
   link_directories(${MX_LIB_DIR})
   list(APPEND SOURCE_LIST ${SRC_XDL_MX_BACKEND})
   IF (NOT USE_GPU)
      list(REMOVE_ITEM SOURCE_LIST "backend/mxnet/backend_gpu_op.cc")       
   ENDIF()
ENDIF()

list(APPEND SOURCE_LIST ${SRC_XDL_CORE_PS_OPS} ${SRC_XDL_CORE_PS_MODEL_SERVER_OPS})

# proto
file(GLOB_RECURSE PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/proto/*.proto)
set(proto_gen_folder "${PROJECT_BINARY_DIR}/xdl/core/proto")
protobuf_generate_cpp_py(${proto_gen_folder} PROTO_SRCS PROTO_HDRS xdl_core_proto_python "${PROJECT_SOURCE_DIR}/xdl/core" "proto" ${PROTOS})
add_library(xdl_core_proto OBJECT ${PROTO_SRCS})

IF (BUILD_SHARED)
   add_library(xdl_core SHARED ${SOURCE_LIST} $<TARGET_OBJECTS:xdl_core_proto>)
   target_link_libraries(xdl_core libprotobuf ${PS_LIBRARYS} ${BACKEND_LIB} python2.7 cudart)
ELSE()
   add_library(xdl_core STATIC ${SOURCE_LIST} $<TARGET_OBJECTS:xdl_core_proto>)
ENDIF()
