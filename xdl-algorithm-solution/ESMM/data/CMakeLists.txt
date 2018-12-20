project(sample_generator)
cmake_minimum_required (VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -std=c++1y -Wcpp -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -g -pipe -fopenmp -DNDEBUG")

# GOOGLE PROTOBUF
set(PROTOBUF_VERSION "protobuf-3.5.0")
set(PROTOBUF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/src)
set(PROTOBUF_PROTOC_PATH ${PROJECT_BINARY_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/cmake/protoc)
set(PROTOBUF_PROTOC_EXECUTABLE protoc)
set(PROTOBUF_LIBRARIES Tricky)
set(PROTOBUF_IMPORT_DIRS ${PROTOBUF_ROOT}/src/)

add_subdirectory(${PROTOBUF_ROOT})


file(GLOB_RECURSE proto_files "proto/*.proto")
message(STATUS "protofiles:" ${proto_files})
include(${PROJECT_SOURCE_DIR}/cmake/ProtoBuf.cmake)
set(proto_gen_folder "${PROJECT_BINARY_DIR}/proto")
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python "${PROJECT_SOURCE_DIR}" "proto" ${proto_files})
message(STATUS "protosrcs: ${proto_srcs}")
message(STATUS "include:" ${PROTOBUF_INCLUDE_DIR})
add_library(sample_proto OBJECT ${proto_srcs})

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/src/
    ${proto_gen_folder}
)

file(GLOB_RECURSE srcs "src/*.cc")
add_executable(sample_generator ${srcs} $<TARGET_OBJECTS:sample_proto>)
target_link_libraries(sample_generator libprotobuf -pthread)
