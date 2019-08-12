cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(sub_string_finder CXX)
add_executable(sub_string_finder sub_string_finder.cpp)

include(${TBB_ROOT}/cmake/TBBBuild.cmake)

# Build Intel TBB with enabled Community Preview Features (CPF).
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS)

find_package(TBB REQUIRED)

# Link Intel TBB imported targets to the executable;
# "TBB::tbb_preview" can be used instead of "${TBB_IMPORTED_TARGETS}".
target_link_libraries(sub_string_finder ${TBB_IMPORTED_TARGETS})
