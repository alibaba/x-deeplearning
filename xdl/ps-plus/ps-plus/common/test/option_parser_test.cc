#include "gtest/gtest.h"
#include "ps-plus/common/option_parser.h"

using ps::CommandLineParameter;
using ps::OptionParser;

TEST(CommandLineParameterTest, CommandLineParameter) {
  auto clp = new CommandLineParameter("run yes no");
  ASSERT_NE(clp, nullptr);
  ASSERT_EQ(clp->getArgc(), 3);
  char **arr = clp->getArgv();
  ASSERT_EQ(std::string(arr[0]), std::string("run"));
  ASSERT_EQ(std::string(arr[1]), std::string("yes"));
  ASSERT_EQ(std::string(arr[2]), std::string("no"));
  delete clp;
}

TEST(OptionParserTest, OptionParser) {
  {
    auto clp = new CommandLineParameter("first yes second 123");
    ASSERT_NE(clp, nullptr);
    ASSERT_EQ(clp->getArgc(), 4);

    char **arr = clp->getArgv();
    auto parser = new OptionParser("my test");
    parser->addOption("fir", "first", "first", OptionParser::OPT_STRING, true);
    parser->addOption("sec", "second", "second", OptionParser::OPT_INT32, true);
    bool res = parser->parseArgs(4, arr);
    ASSERT_EQ(res, true);

    std::string val;
    parser->getOptionValue("first", val);
    ASSERT_EQ(val, "yes");

    int32_t value;
    parser->getOptionValue("second", value);
    ASSERT_EQ(value, 123);

    delete parser;
  }

  {
    auto clp = new CommandLineParameter("first yes second 123 sixth 42");
    ASSERT_NE(clp, nullptr);
    ASSERT_EQ(clp->getArgc(), 6);

    char **arr = clp->getArgv();
    auto parser = new OptionParser("my test");
    parser->addOption("fir", "first", "first", "no");
    parser->addOption("sec", "second", "second", std::string("yes"));
    parser->addOption("thi", "third", "third", int32_t(3));
    parser->addOption("fou", "fourth", "fourth", uint32_t(4));
    parser->addOption("fif", "fifth", "fifth", true);
    parser->addOption("six", "sixth", "sixth", OptionParser::STORE, OptionParser::OPT_UINT32, true);
    bool res = parser->parseArgs(6, arr);
    ASSERT_EQ(res, true);

    std::string val;
    parser->getOptionValue("first", val);
    ASSERT_EQ(val, "yes");

    parser->getOptionValue("second", val);
    ASSERT_EQ(val, std::string("123"));

    int32_t value;
    parser->getOptionValue("third", value);
    ASSERT_EQ(value, 3);

    uint32_t uval;
    parser->getOptionValue("fourth", uval);
    ASSERT_EQ(uval, 4);

    bool bval;
    parser->getOptionValue("fifth", bval);
    ASSERT_EQ(bval, true);

    parser->getOptionValue("sixth", uval);
    ASSERT_EQ(uval, 42);

    delete parser;
  }
}
