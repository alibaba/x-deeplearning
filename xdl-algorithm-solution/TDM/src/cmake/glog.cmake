set(GLOG_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog)
add_subdirectory(${GLOG_ROOT_DIR} third_party/glog)
set(GLOG_INCLUDE_DIRS ${GLOG_ROOT_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/third_party/glog)
