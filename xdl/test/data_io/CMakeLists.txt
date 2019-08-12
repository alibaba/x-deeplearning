aux_source_directory(op SRC_CORE_LIB_TEST)

xdl_add_test_exclude( . ".cc" "global_scheduler_test" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
xdl_add_test( op/feature_op ".cc" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
xdl_add_test( parser ".cc" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
xdl_add_test( packer ".cc" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
xdl_add_test( merger ".cc" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
xdl_add_test( fs ".cc" ${XDL_IO_LIB} ${XDL_CORE_LIB} gtest gtest_main test_util)
