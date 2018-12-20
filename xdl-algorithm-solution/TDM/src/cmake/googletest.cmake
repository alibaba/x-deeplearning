set(GTEST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest)
add_subdirectory(${GTEST_ROOT_DIR} third_party/googletest)
set(GTEST_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include")
