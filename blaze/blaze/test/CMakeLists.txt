#------------------------------------
# Test batching module
#------------------------------------
blaze_add_test(batching ".cc" blaze gtest)

#------------------------------------
# Test common module
#------------------------------------
blaze_add_test(common ".cc" blaze gtest)

#------------------------------------
# Test operator module
#------------------------------------
blaze_add_test(operator ".cc" blaze gtest)
blaze_add_test(operator/op ".cc" blaze gtest)
blaze_add_test(operator/sparse_op/cross ".cc" blaze gtest)
blaze_add_test(operator/sparse_op/embedding ".cc" blaze gtest)
blaze_add_test(operator/fused_op ".cc" blaze gtest)

#------------------------------------
# Test math module
#------------------------------------
blaze_add_test(math ".cc" blaze gtest)
blaze_add_test(math/elementwise ".cc" blaze gtest)

if (USE_CUDA)
blaze_add_cuda_test(math ".cu" blaze gtest)
endif()

#------------------------------------
# Test optimizer module
#------------------------------------
blaze_add_test(optimizer ".cc" blaze gtest)
blaze_add_test(optimizer/passes ".cc" blaze gtest)

#------------------------------------
# Test graph module
#------------------------------------
blaze_add_test(graph ".cc" blaze gtest)
blaze_add_test(graph/observer ".cc" blaze gtest)
blaze_add_test(graph/pattern ".cc" blaze gtest)
blaze_add_test(graph/pattern/in_order ".cc" blaze gtest)
blaze_add_test(graph/pattern/in_parallel ".cc" blaze gtest)
blaze_add_test(graph/transform ".cc" blaze gtest)

#------------------------------------
# Test model importer module
#------------------------------------
blaze_add_test(model_importer ".cc" blaze gtest)

#------------------------------------
# Test store module
#------------------------------------
blaze_add_test(store ".cc" blaze gtest)
blaze_add_test(store/quick_embedding ".cc" blaze gtest)

#------------------------------------
# Test API Module
#------------------------------------
blaze_add_test(api ".cc" blaze gtest)

# copy utest data file
add_custom_target(utest_data ALL DEPENDS)
add_custom_command(TARGET utest_data
    POST_BUILD
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/utest_data ${CMAKE_BINARY_DIR}/blaze/test/ COMMENT "copy utest data")
