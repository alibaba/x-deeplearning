# Generate swig python binding
add_custom_command(
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/dist_tree_api.py"
             "${CMAKE_CURRENT_SOURCE_DIR}/dist_tree_api_wrap.cxx"
      COMMAND "swig"
      ARGS -c++ -python "${CMAKE_CURRENT_SOURCE_DIR}/dist_tree_api.i"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dist_tree_api.i
      COMMENT "Generate swig python bindings for dist_tree_api.i"
      VERBATIM)
add_custom_target(DistTreeSwigGen ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dist_tree_api_wrap.cxx")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_library(dist_tree_api SHARED dist_tree_api_wrap.cxx dist_tree.cc)
target_link_libraries(dist_tree_api dist_tree  rt)

set(TAGET_SHARED_OBJ "${CMAKE_CURRENT_SOURCE_DIR}/../dist_tree/_dist_tree_api.so")
add_custom_command(
  OUTPUT ${TAGET_SHARED_OBJ}
  COMMAND ${CMAKE_COMMAND} -E copy  $<TARGET_FILE:dist_tree_api> ${TAGET_SHARED_OBJ}
  COMMAND ${CMAKE_COMMAND} -E copy  $<TARGET_FILE:dist_tree> ${CMAKE_CURRENT_SOURCE_DIR}/../dist_tree
  DEPENDS dist_tree_api
  VERBATIM)
add_custom_target(dist_tree_copyso ALL DEPENDS ${TAGET_SHARED_OBJ})
