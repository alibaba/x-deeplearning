include_directories(
  ${PROJECT_SOURCE_DIR}/tdm-serving/api
  ${PROJECT_BINARY_DIR}/proto
)

set(UTEST_SOURCE_FILES tdm_example.cpp
                       tdm_with_filter_example.cpp)
tdm_serving_add_executable("${UTEST_SOURCE_FILES}" tdm_serving)

add_custom_target(examples_test_data ALL DEPENDS)
add_custom_command(TARGET examples_test_data
  POST_BUILD
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${CMAKE_BINARY_DIR}/examples/ COMMENT "copy example test data")
