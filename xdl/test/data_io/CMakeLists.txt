aux_source_directory(op SRC_CORE_LIB_TEST)

xdl_add_test( . ".cc" xdl_io xdl_core rdkafka++ jsoncpp glog libprotobuf gtest gtest_main test_util)
xdl_add_test( op/feature_op ".cc" xdl_io xdl_core rdkafka++ jsoncpp glog libprotobuf gtest gtest_main)
xdl_add_test( parser ".cc" xdl_io xdl_core glog rdkafka++ jsoncpp libprotobuf gtest gtest_main)
xdl_add_test( packer ".cc" xdl_io xdl_core glog jsoncpp libprotobuf gtest gtest_main)
xdl_add_test( merger ".cc" xdl_io xdl_core glog jsoncpp libprotobuf gtest gtest_main)
xdl_add_test( fs ".cc" xdl_io xdl_core glog jsoncpp libprotobuf gtest gtest_main test_util)
