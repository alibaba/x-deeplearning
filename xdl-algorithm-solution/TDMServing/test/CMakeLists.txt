#gtest
include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           release-1.8.0
  SOURCE_DIR        "${PROJECT_BINARY_DIR}/gtest/src"
  BINARY_DIR        "${PROJECT_BINARY_DIR}/gtest/build"
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
ExternalProject_Get_Property(googletest BINARY_DIR)
ExternalProject_Get_Property(googletest SOURCE_DIR)
set(GTEST_LIB ${BINARY_DIR}/googlemock/libgmock.a
              ${BINARY_DIR}/googlemock/libgmock_main.a
              ${BINARY_DIR}/googlemock/gtest/libgtest.a
              ${BINARY_DIR}/googlemock/gtest/libgtest_main.a
)
set(GTEST_INCLUDE_PATH
  ${SOURCE_DIR}/googletest/include
  ${SOURCE_DIR}/googlemock/include
)

include_directories(
  ${GTEST_INCLUDE_PATH}
)

# test lib
file(GLOB_RECURSE SOURCE
    "index/mock_index.cpp"
    "index/tree/tree_test_util.cpp"
    "model/mock_model.cpp"
)
add_library(test_lib SHARED ${SOURCE})
add_dependencies(test_lib googletest)
add_dependencies(test_lib tdm_serving)

# test exec
set(UTEST_SOURCE_FILES index/index_conf_test.cpp
                       index/index_unit_test.cpp
                       index/index_manager_test.cpp
                       index/tree/tree_test.cpp
                       index/tree/tree_index_conf_test.cpp
                       index/tree/tree_index_test.cpp
                       index/tree/tree_searcher_test.cpp
                       model/model_conf_test.cpp
                       model/model_unit_test.cpp
                       model/model_manager_test.cpp
                       model/blaze/blaze_model_conf_test.cpp
                       model/blaze/blaze_model_test.cpp
                       biz/filter_manager_test.cpp)

tdm_serving_add_test("${UTEST_SOURCE_FILES}" test_lib tdm_serving ${GTEST_LIB})

# test data
add_custom_target(test_data ALL DEPENDS)
add_custom_command(TARGET test_data
  POST_BUILD
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${CMAKE_BINARY_DIR}/test/ COMMENT "copy test data")
