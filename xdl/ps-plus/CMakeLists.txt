cmake_minimum_required(VERSION 2.8)
project(ps-plus)

if (DEBUG)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++1y -D_GLIBCXX_USE_CXX11_ABI=0")
else ()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++1y -g -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0")
endif ()

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

include_directories(.)

link_directories(${PROJECT_BINARY_DIR}/third_party/googletest/)
link_directories(${PROJECT_BINARY_DIR}/third_party/zookeeper-client/)
link_directories(${PROJECT_BINARY_DIR}/third_party/glog/)

include_directories(${PROJECT_SOURCE_DIR}/third_party/glog/)
include_directories(${PROJECT_SOURCE_DIR}/third_party/hdfs/)

aux_source_directory(ps-plus/service/seastar/lib SEASTAR_LIB)
add_library(seastar_service STATIC ${SEASTAR_LIB})

set(SEASTAR_LIBRARYS -Wl,--whole-archive seastar_service ps_network_static seastar -Wl,--no-whole-archive -L/usr/local/lib64/boost -lboost_timer -lboost_chrono  -laio -lboost_program_options -lboost_system -lboost_filesystem -lstdc++ -lm -lboost_thread -lcryptopp -lrt -lgnutls -lgnutlsxx -llz4 -ldl -lgcc_s -lunwind -lhwloc -lnuma -lpciaccess -lxml2 -lz -lcares-seastar)

SET(PLUGINS )
SET(PLUGINS_DEPENDENCY )
if (USE_HDFS OR NOT DEFINED USE_HDFS)
  SET(PLUGINS ${PLUGINS} ps_plugin_hdfs)
endif()

set(LIBRARYS  -Wl,--whole-archive ps_common ps_server ps_model_server ps_client ps_scheduler libhashtable.a libzookeeper.a libevent_core.a ${PLUGINS} glog -Wl,--no-whole-archive ${PLUGINS_DEPENDENCY} ${SEASTAR_LIBRARYS})

aux_source_directory(ps-plus/common COMMON)
aux_source_directory(ps-plus/common/initializer COMMON_INITIALIZER)
aux_source_directory(ps-plus/common/initializer/random COMMON_INITIALIZER_RANDOM)
aux_source_directory(ps-plus/common/file_system COMMON_FILESYSTEM)
aux_source_directory(ps-plus/message MESSAGE)
aux_source_directory(ps-plus/message/test MESSAGE_TEST)
aux_source_directory(ps-plus/server SERVER)
aux_source_directory(ps-plus/server/udf SERVER_UDF)
aux_source_directory(ps-plus/client CLIENT)
aux_source_directory(ps-plus/client/partitioner CLIENT_PARTITIONER)
aux_source_directory(ps-plus/scheduler SCHEDULER)
aux_source_directory(ps-plus/scheduler/test SCHEDULER_TEST)
aux_source_directory(ps-plus/main MAIN)
aux_source_directory(ps-plus/model_server MODEL_SERVER)
aux_source_directory(ps-plus/model_server/test MODEL_SERVER_TEST)
aux_source_directory(ps-plus/tool CLIENT_TOOL)
aux_source_directory(ps-plus/profiler PROFILER)
aux_source_directory(ps-plus/common/test COMMON_TEST)
aux_source_directory(ps-plus/common/initializer/test COMMON_INITIALIZER_TEST)
aux_source_directory(ps-plus/server/test SERVER_TEST)
aux_source_directory(ps-plus/server/udf/test SERVER_UDF_TEST)
aux_source_directory(ps-plus/client/test CLIENT_TEST)
aux_source_directory(ps-plus/client/partitioner/test CLIENT_PARTITIONER_TEST)
aux_source_directory(../test/util SRC_TEST_UTIL)

aux_source_directory(ps-plus/plugins/hdfs PLUGINS_HDFS)

add_library(ps_common STATIC ${COMMON} ${COMMON_INITIALIZER} ${COMMON_INITIALIZER_RANDOM} ${COMMON_FILESYSTEM} ${MESSAGE})
add_library(ps_server STATIC ${SERVER} ${SERVER_UDF})
add_library(ps_model_server STATIC ${MODEL_SERVER})
add_library(ps_scheduler STATIC ${SCHEDULER})
add_library(ps_client STATIC ${CLIENT} ${CLIENT_PARTITIONER})
add_library(ps_plugin_hdfs STATIC ${PLUGINS_HDFS})

add_executable(ps ${MAIN})
add_executable(tool ${CLIENT_TOOL})

# tests
add_executable(ps_common_test ${COMMON_TEST} ${COMMON_INITIALIZER_TEST} ${SRC_TEST_UTIL})
add_executable(ps_model_server_test ${MODEL_SERVER_TEST})
add_executable(ps_message_test ${MESSAGE_TEST})
add_executable(ps_server_test ${SERVER_TEST} ${SERVER_UDF_TEST})
add_executable(ps_client_test ${CLIENT_TEST} ${CLIENT_PARTITIONER_TEST})
add_executable(ps_scheduler_test ${SCHEDULER_TEST})

# profiler
add_executable(ps_profiler ${PROFILER})

target_link_libraries(ps ${LIBRARYS} libjemalloc.a)
target_link_libraries(tool ${LIBRARYS} libjemalloc.a)
target_link_libraries(ps_common_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_message_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_model_server_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_server_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_client_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_scheduler_test ${LIBRARYS} gtest gtest_main libjemalloc.a)
target_link_libraries(ps_profiler ${LIBRARYS} libjemalloc.a)

enable_testing()
add_test(NAME ps_common_test COMMAND ps_common_test)
add_test(NAME ps_message_test COMMAND ps_message_test)
add_test(NAME ps_model_server_test COMMAND ps_model_server_test)
add_test(NAME ps_server_test COMMAND ps_server_test)
add_test(NAME ps_client_test COMMAND ps_client_test)
add_test(NAME ps_scheduler_test COMMAND ps_scheduler_test)

if (APPLE)
  set_target_properties(ps PROPERTIES LINK_FLAGS "-rpath /usr/local/lib")
elseif (UNIX)
  set_target_properties(ps PROPERTIES LINK_FLAGS "")
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/ps-plus DESTINATION include FILES_MATCHING PATTERN "*.h")
