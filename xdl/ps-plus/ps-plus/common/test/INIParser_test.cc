#include <fstream>

#include "gtest/gtest.h"
#include "ps-plus/common/INIParser.h"

using ps::INIParser;

TEST(INIParserTest, INIParser) {
  const char *file_name = "/tmp/test.ini";
  std::ofstream fp(file_name);
  fp << "[fruit]\n";
  fp << "a = apple\n";
  fp << "b = banana\n";
  fp << "c = coconut\n";
  fp << "d = 321\n";
  fp << "f = true\n";
  fp.close();

  auto parser = new INIParser(file_name);
  ASSERT_NE(parser, nullptr);

  parser->dump();

  std::string value = parser->get_string("fruit", "a", "");
  ASSERT_EQ(value, "apple");

  int val = parser->get_int("fruit", "e", -1);
  ASSERT_EQ(val, -1);

  val = parser->get_int(std::string("fruit"), "d", -1);
  ASSERT_EQ(val, 321);

  unsigned uvalue = parser->get_unsigned("fruit", "d", 0);
  ASSERT_EQ(uvalue, 321);

  uvalue = parser->get_unsigned(std::string("fruit"), "e", -1);
  ASSERT_EQ(uvalue, -1);

  bool res = parser->get_bool(std::string("fruit"), std::string("f"), false);
  ASSERT_EQ(res, true);

  res = parser->get_bool("fruit", "g", true);
  ASSERT_EQ(res, true);

  std::string content;
  content = parser->get_section(std::string("fruit"));
  ASSERT_GT(content.length(), 1);

  const char *section = parser->get_section("free");
  ASSERT_EQ(section, nullptr);

  delete parser;
}
