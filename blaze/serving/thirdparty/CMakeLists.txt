include(ExternalProject)

ExternalProject_Add(libevent
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libevent/
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libevent/
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/libevent/
  CMAKE_ARGS -DBuildExamples=OFF -DEVENT__DISABLE_OPENSSL=ON 
             -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/libevent/
)

ExternalProject_Get_Property(libevent install_dir)

message("libevent install_dir = ${install_dir}")
set(LIBEVENT_INCLUDE_DIRS ${install_dir}/include/ PARENT_SCOPE)

add_library(thirdparty_libevent STATIC IMPORTED GLOBAL)
set_target_properties(thirdparty_libevent 
                      PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libevent.a)
add_dependencies(thirdparty_libevent libevent)

#find_package(libevent HINTS ${install_dir})

