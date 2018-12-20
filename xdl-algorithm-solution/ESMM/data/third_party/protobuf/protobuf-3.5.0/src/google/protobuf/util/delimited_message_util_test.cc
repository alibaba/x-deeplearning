/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Adapted from the patch of kenton@google.com (Kenton Varda)
// See https://github.com/google/protobuf/pull/710 for details.

#include <google/protobuf/util/delimited_message_util.h>

#include <sstream>

#include <google/protobuf/test_util.h>
#include <google/protobuf/unittest.pb.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {
namespace util {

TEST(DelimitedMessageUtilTest, DelimitedMessages) {
  std::stringstream stream;

  {
    protobuf_unittest::TestAllTypes message1;
    TestUtil::SetAllFields(&message1);
    EXPECT_TRUE(SerializeDelimitedToOstream(message1, &stream));

    protobuf_unittest::TestPackedTypes message2;
    TestUtil::SetPackedFields(&message2);
    EXPECT_TRUE(SerializeDelimitedToOstream(message2, &stream));
  }

  {
    bool clean_eof;
    io::IstreamInputStream zstream(&stream);

    protobuf_unittest::TestAllTypes message1;
    clean_eof = true;
    EXPECT_TRUE(ParseDelimitedFromZeroCopyStream(&message1,
        &zstream, &clean_eof));
    EXPECT_FALSE(clean_eof);
    TestUtil::ExpectAllFieldsSet(message1);

    protobuf_unittest::TestPackedTypes message2;
    clean_eof = true;
    EXPECT_TRUE(ParseDelimitedFromZeroCopyStream(&message2,
        &zstream, &clean_eof));
    EXPECT_FALSE(clean_eof);
    TestUtil::ExpectPackedFieldsSet(message2);

    clean_eof = false;
    EXPECT_FALSE(ParseDelimitedFromZeroCopyStream(&message2,
        &zstream, &clean_eof));
    EXPECT_TRUE(clean_eof);
  }
}

}  // namespace util
}  // namespace protobuf
}  // namespace google
