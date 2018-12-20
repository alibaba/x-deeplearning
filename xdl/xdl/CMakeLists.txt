file(GLOB_RECURSE proto_files "proto/*.proto")
message(STATUS "protofiles: ${proto_files}")

link_directories(
    ${PROJECT_BINARY_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/cmake/
)

include_directories(
        ${PROJECT_BINARY_DIR}/
        ${PROJECT_SOURCE_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/src/
        )

include(${PROJECT_SOURCE_DIR}/cmake/ProtoBuf.cmake)
set(proto_gen_folder "${PROJECT_BINARY_DIR}/xdl/proto")
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python "${PROJECT_SOURCE_DIR}/xdl" "proto" ${proto_files})
message(STATUS "protosrcs: ${proto_srcs}")
message(STATUS "include: ${PROTOBUF_INCLUDE_DIR}")
add_library(xdl_proto OBJECT ${proto_srcs})

# backend
IF (TF_BACKEND)
   include_directories(${TF_INCLUDE_DIR})                   
   link_directories(${TF_LIB_DIR})
ENDIF()

IF (MXNET_BACKEND)
   include_directories(${MX_INCLUDE_DIR})
   link_directories(${MX_LIB_DIR})
ENDIF()

add_subdirectory(core)
add_subdirectory(data_io)
add_subdirectory(python)

