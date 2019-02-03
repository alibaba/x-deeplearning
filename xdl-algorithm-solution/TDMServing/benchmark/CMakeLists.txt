# bench exec
set(BENCH_SOURCE_FILES tdm_benchmark.cpp)
tdm_serving_add_executable("${BENCH_SOURCE_FILES}" tdm_serving)

# bench data
add_custom_target(bench_data ALL DEPENDS)
add_custom_command(TARGET bench_data
  POST_BUILD
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/bench_data ${CMAKE_BINARY_DIR}/benchmark/ COMMENT "copy bench data")
