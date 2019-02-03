set(UTEST_SOURCE_FILES tdm_evaluation.cpp)
tdm_serving_add_executable("${UTEST_SOURCE_FILES}" tdm_serving)

add_custom_target(eval_data ALL DEPENDS)
add_custom_command(TARGET eval_data
  POST_BUILD
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/eval_data ${CMAKE_BINARY_DIR}/evaluation/ COMMENT "copy evaluation data")
