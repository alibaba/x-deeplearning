cmake_minimum_required (VERSION 3.4)
project (PSService)

if (USE_STATISTICS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_STATISTICS")
endif ()

set (SEASTAR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (SEASTAR_BINARY_DIR ${SEASTAR_SOURCE_DIR}/build/release)
set (ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${SEASTAR_BINARY_DIR}")

find_package (PkgConfig REQUIRED)
pkg_search_module (SEASTAR REQUIRED seastar)

# 'libseastar.a' is compiled with '-fPIC' so that it can be
# linked into a shared library. After linking, the '-fPIC'
# flag is not needed any more, so we remove it.
#list (REMOVE_ITEM SEASTAR_CFLAGS -fPIC)
#list (REMOVE_ITEM SEASTAR_LDFLAGS -fPIC)

# The following link flags should only be used when linking
# the static libraries into the shared library, and should
# not be exposed publicly (otherwise, it will cause duplicate
# definitions of global data objects). So we remove it, and
# add them explicitly when needed.
list (REMOVE_ITEM SEASTAR_LDFLAGS -Wl,--whole-archive,-lseastar,--no-whole-archive)
list (REMOVE_ITEM SEASTAR_LDFLAGS -lcares-seastar)

set (PS_NETWORK_SOURCE_FILES)
file (GLOB PS_SERVICE_SOURCE_FILES   ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file (GLOB PS_QUEUE_HUB_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../core/ps_queue_hub/*.cc)
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
