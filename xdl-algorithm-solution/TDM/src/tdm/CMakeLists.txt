enable_testing()

add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tree.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/tree.pb.h"
             "${CMAKE_CURRENT_BINARY_DIR}/store_kv.pb.cc"

      COMMAND 
      "${PROTOC_BIN}" --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/tree.proto
      COMMAND
      "${PROTOC_BIN}" --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/store_kv.proto
      COMMAND
      "${PROTOC_BIN}" --python_out  ${CMAKE_CURRENT_SOURCE_DIR}/../python/cluster -I ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/tree.proto

      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tree.proto
      COMMENT "Running C++ protocol buffer compiler on tree.proto"
      VERBATIM)
add_custom_target(GenPb ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tree.pb.cc")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(dist_tree SHARED
            dist_tree.cc
            cache.cc
            store.cc
            local_store.cc
            bitmap.cc
            common.cc
            ${CMAKE_CURRENT_BINARY_DIR}/tree.pb.cc
            ${CMAKE_CURRENT_BINARY_DIR}/store_kv.pb.cc)

target_link_libraries(dist_tree ${CMAKE_THREAD_LIBS_INIT}  ${XDL_LIB})

add_library(selector SHARED selector.cc brother_selector.cc layerwise_selector.cc tdm_op.cc tdm_predict_op.cc)
target_link_libraries(selector dist_tree)

add_executable(dist_tree_checker dist_tree_checker.cc)
target_link_libraries(dist_tree_checker selector)

# Add test
 add_executable(dist_tree_test dist_tree_test.cc bitmap_test.cc store_test.cc)
 target_link_libraries(dist_tree_test selector gtest gtest_main)

 add_executable(tdm_op_test tdm_op_test.cc)
 target_link_libraries(tdm_op_test selector gtest gtest_main)
 
