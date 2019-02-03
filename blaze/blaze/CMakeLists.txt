file(GLOB_RECURSE SOURCE
    "common/*.cc"
    "operator/*.cc"
    "math/*.cc"
    "graph/*.cc"
    "optimizer/*.cc"
    "model_importer/*.cc"
    "batching/*.cc"
    "scheduler/*.cc"
    "store/*.cc"
    "api/*.cc"
)

set(proto_gen_folder "${PROJECT_BINARY_DIR}/blaze/proto")
file(GLOB_RECURSE proto_files "proto/*.proto")
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python
       "${PROJECT_SOURCE_DIR}/blaze" "proto" ${proto_files})
list(APPEND SOURCE ${proto_srcs})

set(model_importer_proto_gen_folder "${PROJECT_BINARY_DIR}/blaze/model_importer")
file(GLOB_RECURSE model_importer_proto_files "model_importer/*.proto")
protobuf_generate_cpp_py(${model_importer_proto_gen_folder} model_importer_proto_srcs model_importer_proto_hdrs
    model_importer_proto_python
    "${PROJECT_SOURCE_DIR}/blaze" "model_importer" ${model_importer_proto_files})
list(APPEND SOURCE ${model_importer_proto_srcs})

install(FILES ${proto_gen_folder}/blaze_pb2.py DESTINATION ${PROJECT_SOURCE_DIR}/binding/python/pyblaze/)
install(FILES ${model_importer_proto_gen_folder}/ulf_pb2.py DESTINATION ${PROJECT_SOURCE_DIR}/binding/python/pyblaze/)

if (USE_CUDA)
file(GLOB_RECURSE CUDA_SOURCE "operator/*.cu" "math/*.cu")
blaze_cuda_compile(cuda_objs ${CUDA_SOURCE})
list(APPEND SOURCE ${cuda_objs})
endif()

# source-code level link protobuf and onnx and zlib
include(${PROJECT_SOURCE_DIR}/thirdparty/protobuf/${PROTOBUF_VERSION}/cmake/libprotobuf_src.cmake)
list(APPEND SOURCE ${libprotobuf_files} ${libprotobuf_lite_files})
include(${PROJECT_SOURCE_DIR}/thirdparty/onnx/${ONNX_VERSION}/cmake/onnx_src.cmake)
list(APPEND SOURCE ${LIB_ONNX_SOURCE})
include(${PROJECT_SOURCE_DIR}/thirdparty/zlib/zlib-1.2.8/zlib.cmake)
list(APPEND SOURCE ${LIB_ZLIB_SOURCE})

if (COMPILE_STATIC)
add_library(blaze STATIC ${SOURCE})
else()
add_library(blaze SHARED ${SOURCE})
endif()

if (SYMBOL_EXPORT_CTL)
SET(CMAKE_SHARED_LINKER_FLAGS -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libinfo.map)
endif()

if (USE_MKL)
target_link_libraries(blaze 
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 pthread m)
else()
target_link_libraries(blaze openblas)
endif()

if (USE_CUDA)
target_link_libraries(blaze -lcudart -lcublas -lcurand -lnvrtc -lcuda -lcudnn)
endif()

if (COVERAGE)
set_target_properties(blaze PROPERTIES COMPILE_FLAGS "-g --coverage")
endif()

if (SYMBOL_EXPORT_CTL)
else()
add_subdirectory(test)
endif()

install(TARGETS blaze DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/api/cpp_api/predictor.h DESTINATION include)
