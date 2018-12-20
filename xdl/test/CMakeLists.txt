add_custom_target(utest ALL DEPENDS)
add_custom_command(TARGET utest
  POST_BUILD
  COMMAND cp -r ${PROJECT_SOURCE_DIR}/test/test_data/* ${CMAKE_BINARY_DIR}/test/
  COMMENT "copy ${PROJECT_SOURCE_DIR}/test/test_data/ ${CMAKE_BINARY_DIR}/test/")

add_custom_target(write_path ALL DEPENDS)
add_custom_command(TARGET write_path
  PRE_BUILD
  COMMAND echo 'const char *XDL_HADOOP_PATH = \"${PROJECT_SOURCE_DIR}/test/binary/hadoop-2.8.5/\"\;' > ${PROJECT_SOURCE_DIR}/test/util/hdfs_define.h && echo 'const char *XDL_ZOOKEEPER_PATH = \"${PROJECT_SOURCE_DIR}/test/binary/zookeeper-3.4.13/\"\;' > ${PROJECT_SOURCE_DIR}/test/util/zookeeper_define.h)

include_directories(${PROTOBUF_ROOT}/src)
link_directories(${PROJECT_SOURCE_DIR}/third_party/jsoncpp/lib64/)
link_directories(${PROJECT_BINARY_DIR}/third_party/protobuf/${PROTOBUF_VERSION}/cmake/)
link_directories(${PROJECT_BINARY_DIR}/third_party/glog/)

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

aux_source_directory(util SRC_TEST_UTIL)
add_library(test_util STATIC ${SRC_TEST_UTIL})
