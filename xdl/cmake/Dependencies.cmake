# zookeeper
set(ZOOKEEPER_CLIENT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zookeeper-client")
add_subdirectory(${ZOOKEEPER_CLIENT_ROOT})

# gtest
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARIES gtest gtest_main)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)
add_subdirectory(${GTEST_ROOT})
find_package(GTest REQUIRED)

# protobuf
set(PROTOBUF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/src)
set(PROTOBUF_PROTOC_PATH ${PROJECT_BINARY_DIR}/third_party/protobuf/cmake/protoc)
set(PROTOBUF_PROTOC_EXECUTABLE protoc)
set(PROTOBUF_LIBRARIES Tricky)
set(PROTOBUF_IMPORT_DIRS ${PROTOBUF_ROOT}/src/)
add_subdirectory(${PROTOBUF_ROOT}/cmake/)

# eigen
# set(EIGEN_MPL2_ONLY 1)
set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
include_directories(${Eigen3_DIR})

# glog
set(GLOG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog")
add_subdirectory(${GLOG_ROOT})
include_directories(${PROJECT_BINARY_DIR}/third_party/glog/)
include_directories(${PROJECT_SOURCE_DIR}/third_party/glog/src/)

# librdkafka
option(RDKAFKA_BUILD_TESTS off)
set(RDKAFKA_BUILD_TESTS off)
set(LIB_RDKAFKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/librdkafka")
add_subdirectory(${LIB_RDKAFKA_ROOT})

# pybind11
set(PYBIND11_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11")
include_directories(${PYBIND11_ROOT}/include/)

# hdfs
set(HDFS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hdfs")
include_directories(${HDFS_ROOT}/)

# seastar
set(SEASTAR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seastar")
execute_process(COMMAND bash "build.sh"
                WORKING_DIRECTORY ${SEASTAR_ROOT}/service/build_script)
include_directories(${SEASTAR_ROOT} ${SEASTAR_ROOT}/fmt ${SEASTAR_ROOT}/c-ares)
link_directories(${SEASTAR_ROOT}/build/release ${SEASTAR_ROOT}/service/build_script)

# libevent
option(EVENT__DISABLE_OPENSSL on)
set(EVENT__DISABLE_OPENSSL on)
option(EVENT__DISABLE_TESTS on)
set(EVENT__DISABLE_TESTS on)
set(LIBEVENT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libevent")
add_subdirectory(${LIBEVENT_ROOT})
include_directories(${PROJECT_BINARY_DIR}/third_party/libevent/include/)
include_directories(${PROJECT_SOURCE_DIR}/third_party/libevent/include/)
link_directories(${PROJECT_BINARY_DIR}/third_party/libevent/lib/)        
