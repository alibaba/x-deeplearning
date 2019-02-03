add_subdirectory(thirdparty)

message("libevent_include_dir = ${LIBEVENT_INCLUDE_DIRS}")

include_directories(
  ${LIBEVENT_INCLUDE_DIRS}
  ${PROJECT_BINARY_DIR}/serving/frame/
  ${PROJECT_SOURCE_DIR}/blaze/api/cpp_api/
)

file(GLOB SERVER_SRC
  "model/model.cc"
  "model/model_manager.cc"
  "frame/process.cc"
  "frame/main.cc"
)

protobuf_generate_cpp_py("${CMAKE_CURRENT_BINARY_DIR}/frame" proto_srcs proto_hdrs proto_python 
  "${CMAKE_CURRENT_SOURCE_DIR}" "frame" "./frame/predict.proto")

list(APPEND SERVER_SRC ${proto_srcs})

add_executable(server ${SERVER_SRC})
target_link_libraries(server blaze thirdparty_libevent)

install(FILES ${PROJECT_BINARY_DIR}/serving/server
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    WORLD_EXECUTE WORLD_READ GROUP_EXECUTE GROUP_READ
        DESTINATION bin)

install(FILES client.py DESTINATION bin)
install(FILES config DESTINATION conf)
