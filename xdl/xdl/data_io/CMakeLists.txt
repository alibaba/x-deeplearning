file(GLOB_RECURSE SRC_XDL_IO_LIB
    "*.cc"
    "op/*.cc"
    "fs/*.cc"
    "parser/*.cc"
    "log/*.cc"
)

include_directories(${PROJECT_SOURCE_DIR}/ps-plus/third_party/jsoncpp/include)
link_directories(${PROJECT_BINARY_DIR}/ps-plus/third_party/jsoncpp/lib64)   

IF (BUILD_SHARED)
    add_library(xdl_io SHARED ${SRC_XDL_IO_LIB} $<TARGET_OBJECTS:xdl_proto>)
    target_link_libraries(xdl_io dl rdkafka++ jsoncpp)
ELSE()
    add_library(xdl_io STATIC ${SRC_XDL_IO_LIB} $<TARGET_OBJECTS:xdl_proto>)
    target_link_libraries(xdl_io dl rdkafka++ jsoncpp)
ENDIF()
