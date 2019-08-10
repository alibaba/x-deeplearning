file(GLOB_RECURSE SRC_XDL_IO_LIB
    "*.cc"
    "op/*.cc"
    "fs/*.cc"
    "parser/*.cc"
)

IF (NOT USE_PS_PLUS)
    list(REMOVE_ITEM SRC_XDL_IO_LIB "${CMAKE_CURRENT_SOURCE_DIR}/global_scheduler.cc")
ENDIF()

IF (BUILD_SHARED)
    add_library(xdl_io SHARED ${SRC_XDL_IO_LIB} $<TARGET_OBJECTS:xdl_proto>)
	target_link_libraries(xdl_io ${XDL_IO_DEPEND_LIB})
ELSE()
    add_library(xdl_io STATIC ${SRC_XDL_IO_LIB} $<TARGET_OBJECTS:xdl_proto>)
    target_link_libraries(xdl_io ${XDL_IO_DEPEND_LIB})
ENDIF()
