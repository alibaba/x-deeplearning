#include "gtest/gtest.h"
#include "ps-plus/common/string_utils.h"

using ps::StringUtils;

TEST(StringUtilsTest, StringUtils) {
  {
    int64_t value;
    bool res = StringUtils::strToInt64("9223372036854775807", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 9223372036854775807);

    res = StringUtils::strToInt64(nullptr, value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToInt64("36893488147419103232", value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToInt64("hello", value);
    ASSERT_EQ(res, false);
  }

  {
    uint64_t value;
    bool res = StringUtils::strToUInt64("18446744073709551615", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 18446744073709551615ul);

    res = StringUtils::strToUInt64(nullptr, value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToUInt64("36893488147419103232", value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToUInt64("hello", value);
    ASSERT_EQ(res, false);
  }

  {
    int32_t value;
    bool res = StringUtils::strToInt32("1048576", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 1048576);

    res = StringUtils::strToInt32(nullptr, value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToInt32("8589934592", value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToInt32("hello", value);
    ASSERT_EQ(res, false);
  }

  {
    uint32_t value;
    bool res = StringUtils::strToUInt32("4294967295", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 4294967295);

    res = StringUtils::strToUInt32(nullptr, value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToUInt32("8589934592", value);
    ASSERT_EQ(res, false);
    res = StringUtils::strToUInt32("hello", value);
    ASSERT_EQ(res, false);
  }

  {
    int16_t value;
    bool res = StringUtils::strToInt16("32767", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 32767);
  }

  {
    uint16_t value;
    bool res = StringUtils::strToUInt16("65535", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 65535);
  }

  {
    int8_t value;
    bool res = StringUtils::strToInt8("127", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 127);
  }

  {
    uint8_t value;
    bool res = StringUtils::strToUInt8("255", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, 255);
  }

  {
    float value;
    bool res = StringUtils::strToFloat("12.3456", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, float(12.3456));
  }

  {
    double value;
    bool res = StringUtils::strToDouble("12.3456789", value);
    ASSERT_EQ(res, true);
    ASSERT_EQ(value, double(12.3456789));
  }

  {
    auto fields = StringUtils::split("123:hello:456", ":", false);
    ASSERT_EQ(fields.size(), 3);
    ASSERT_EQ(fields[0], "123");
    ASSERT_EQ(fields[1], "hello");
    ASSERT_EQ(fields[2], "456");
  }

  {
    std::string value = StringUtils::toString(float(12.3456));
    ASSERT_EQ(value, "12.3456");
    value = StringUtils::toString(double(12.3456789));
    ASSERT_EQ(value, "12.3456789");
    value = StringUtils::toString(int8_t(12));
    ASSERT_EQ(value, "12");
    value = StringUtils::toString(int16_t(-32767));
    ASSERT_EQ(value, "-32767");
    value = StringUtils::toString(int32_t(3456789));
    ASSERT_EQ(value, "3456789");
  }

  {
    std::map<std::string, std::string> dict;
    dict["hello"] = "world";
    dict["123"] = "456";

    std::string value;
    StringUtils::GetValueFromMap(dict, "hello", &value);
    ASSERT_EQ(value, "world");
  }
}
