find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Generate swig python binding
add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/store_api.py
             ${CMAKE_CURRENT_SOURCE_DIR}/store_api_wrap.cxx
      COMMAND swig
      ARGS -c++ -python ${CMAKE_CURRENT_SOURCE_DIR}/store_api.i
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/store_api.i
      COMMENT "Generate swig python bindings for store_api.i"
      VERBATIM)
add_custom_target(StoreSwigGen ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/store_api_wrap.cxx")

add_library(store_api SHARED store_api_wrap.cxx store.cc)
target_link_libraries(store_api dist_tree)

add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../store/store_kv_pb2.py
      COMMAND
      ${PROTOC_BIN} --python_out ${CMAKE_CURRENT_SOURCE_DIR}/../store -I ${PROJECT_SOURCE_DIR}/tdm ${PROJECT_SOURCE_DIR}/tdm/store_kv.proto
      DEPENDS ${PROJECT_SOURCE_DIR}/tdm/store_kv.proto
      COMMENT "Running python protocol buffer compiler on store_kv.proto"
      VERBATIM)
add_custom_target(GenStoreKVPb ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../store/store_kv_pb2.py")

set(TAGET_SHARED_OBJ "${CMAKE_CURRENT_SOURCE_DIR}/../store/_store_api.so")
add_custom_command(
  OUTPUT ${TAGET_SHARED_OBJ}
  COMMAND ${CMAKE_COMMAND} -E copy  $<TARGET_FILE:store_api> ${TAGET_SHARED_OBJ}
  COMMAND ${CMAKE_COMMAND} -E copy  $<TARGET_FILE:dist_tree> ${CMAKE_CURRENT_SOURCE_DIR}/../store
  DEPENDS store_api GenStoreKVPb
  VERBATIM)
add_custom_target(store_copyso ALL DEPENDS ${TAGET_SHARED_OBJ})

add_executable(store_test store.cc store_test.cc)
target_link_libraries(store_test dist_tree)
