cmake_minimum_required (VERSION 3.4)
project (PSService)

set (SEASTAR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (SEASTAR_BINARY_DIR ${SEASTAR_SOURCE_DIR}/build/release)
set (SERVICE_SOURCE_DIR ${SEASTAR_SOURCE_DIR}/service)
set (ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${SEASTAR_BINARY_DIR}")

find_package (PkgConfig REQUIRED)
pkg_search_module (SEASTAR REQUIRED seastar)

# 'libseastar.a' is compiled with '-fPIC' so that it can be
# linked into a shared library. After linking, the '-fPIC'
# flag is not needed any more, so we remove it.
list (REMOVE_ITEM SEASTAR_CFLAGS -fPIC)
list (REMOVE_ITEM SEASTAR_LDFLAGS -fPIC)

# The following link flags should only be used when linking
# the static libraries into the shared library, and should
# not be exposed publicly (otherwise, it will cause duplicate
# definitions of global data objects). So we remove it, and
# add them explicitly when needed.
list (REMOVE_ITEM SEASTAR_LDFLAGS -Wl,--whole-archive,-lseastar,--no-whole-archive)
list (REMOVE_ITEM SEASTAR_LDFLAGS -lcares-seastar)

set (PS_NETWORK_SOURCE_FILES)
file (GLOB PS_SERVICE_SOURCE_FILES   ${SERVICE_SOURCE_DIR}/*.cc)
file (GLOB PS_QUEUE_HUB_SOURCE_FILES ${SEASTAR_SOURCE_DIR}/core/ps_queue_hub/*.cc)
list (APPEND PS_NETWORK_SOURCE_FILES ${PS_SERVICE_SOURCE_FILES})
list (APPEND PS_NETWORK_SOURCE_FILES ${PS_QUEUE_HUB_SOURCE_FILES})

add_library (ps_network_shared SHARED ${PS_NETWORK_SOURCE_FILES})
target_link_libraries (ps_network_shared PRIVATE -Wl,--whole-archive,-lseastar,--no-whole-archive)
target_link_libraries (ps_network_shared PRIVATE -Wl,--whole-archive,-lcares-seastar,--no-whole-archive)
target_link_libraries (ps_network_shared PUBLIC ${SEASTAR_LDFLAGS})
target_compile_options (ps_network_shared PUBLIC ${SEASTAR_CFLAGS})

add_library (ps_network_static STATIC ${PS_NETWORK_SOURCE_FILES})
target_link_libraries (ps_network_static PUBLIC -Wl,--whole-archive,-lseastar,--no-whole-archive)
target_link_libraries (ps_network_static PUBLIC -Wl,--whole-archive,-lcares-seastar,--no-whole-archive)
target_link_libraries (ps_network_static PUBLIC ${SEASTAR_LDFLAGS})
target_compile_options (ps_network_static PUBLIC ${SEASTAR_CFLAGS})

#add_executable (ps_client ${PS_CLIENT_SOURCE_FILES})
#set_target_properties (ps_client PROPERTIES OUTPUT_NAME client)
#target_link_libraries (ps_client PUBLIC -ljemalloc)
#target_link_libraries (ps_client PUBLIC ps_network_shared)

#add_executable (ps_server ${PS_SERVER_SOURCE_FILES})
#set_target_properties (ps_server PROPERTIES OUTPUT_NAME server)
#target_link_libraries (ps_server PUBLIC -ljemalloc)
#target_link_libraries (ps_server PUBLIC ps_network_shared)

#list (REMOVE_ITEM PS_ACTION_TEST_SOURCE_FILES ${SEASTAR_SOURCE_DIR}/../examples/enhanced/server_main.cc)
#list (REMOVE_ITEM PS_ACTION_TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../examples/enhanced/worker_main.cc)

file (GLOB PS_ACTION_TEST_SOURCE_FILES ${SEASTAR_SOURCE_DIR}/action/*.cc)

set (PS_ACTION_TEST_SERVER_SOURCE_FILES)
list (APPEND PS_ACTION_TEST_SERVER_SOURCE_FILES ${SEASTAR_SOURCE_DIR}/action/test/server_main.cc)
list (APPEND PS_ACTION_TEST_SERVER_SOURCE_FILES ${PS_ACTION_TEST_SOURCE_FILES})
add_executable (ps_action_test_server ${PS_ACTION_TEST_SERVER_SOURCE_FILES})
set_target_properties (ps_action_test_server PROPERTIES OUTPUT_NAME ps_action_test_server)
target_link_libraries (ps_action_test_server PUBLIC ps_network_shared)
#target_link_libraries (ps_server PUBLIC -ljemalloc)

set (PS_ACTION_TEST_WORKER_SOURCE_FILES)
list (APPEND PS_ACTION_TEST_WORKER_SOURCE_FILES ${SEASTAR_SOURCE_DIR}/action/test/worker_main.cc)
list (APPEND PS_ACTION_TEST_WORKER_SOURCE_FILES ${PS_ACTION_TEST_SOURCE_FILES})
add_executable (ps_action_test_worker ${PS_ACTION_TEST_WORKER_SOURCE_FILES})
set_target_properties (ps_action_test_worker PROPERTIES OUTPUT_NAME ps_action_test_worker)
target_link_libraries (ps_action_test_worker PUBLIC ps_network_shared)
#target_link_libraries (ps_client PUBLIC -ljemalloc)

