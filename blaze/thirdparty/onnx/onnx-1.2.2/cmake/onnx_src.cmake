set(onnx_source_dir ${PROJECT_SOURCE_DIR}/thirdparty/onnx/${ONNX_VERSION}/)
set(ONNX_ROOT ${onnx_source_dir})
#message(${ROOT_DIR})

SET(ONNX_NAMESPACE "onnx")
SET(MY_ONNX_NAMESPACE "-DONNX_NAMESPACE=onnx")
add_definitions(${MY_ONNX_NAMESPACE})

file(GLOB_RECURSE onnx_src
    "${ONNX_ROOT}/onnx/*.h"
    "${ONNX_ROOT}/onnx/*.cc"
)
file(GLOB_RECURSE onnx_gtests_src
    "${ONNX_ROOT}/onnx/test/c++/*.h"
    "${ONNX_ROOT}/onnx/test/c++/*.cc"
)
list(REMOVE_ITEM onnx_src "${ONNX_ROOT}/onnx/cpp2py_export.cc")
list(REMOVE_ITEM onnx_src ${onnx_gtests_src})

list(APPEND LIB_ONNX_SOURCE ${onnx_src})

# Add Protobuf Files
function(RELATIVE_PROTOBUF_GENERATE_CPP NAME SRCS HDRS ROOT_DIR DEPEND)
  
endfunction()

set(onnx_proto_gen_folder "${ONNX_ROOT}/onnx")
include_directories(${ONNX_ROOT}/onnx
    ${ONNX_ROOT})
set(onnx_proto_files
  "${ONNX_ROOT}/onnx/onnx-operators.proto"
  "${ONNX_ROOT}/onnx/onnx.proto"
)
protobuf_generate_cpp_py(${onnx_proto_gen_folder} onnx_proto_srcs onnx_proto_hdrs onnx_proto_python
    "${PROJECT_SOURCE_DIR}/" "${ONNX_ROOT}/onnx" ${onnx_proto_files})
list(APPEND LIB_ONNX_SOURCE ${onnx_proto_srcs})
