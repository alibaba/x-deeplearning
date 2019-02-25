// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// This file defines a protocol for running the conformance test suite
// in-process.  In other words, the suite itself will run in the same process as
// the code under test.
//
// For pros and cons of this approach, please see conformance.proto.

#ifndef CONFORMANCE_CONFORMANCE_TEST_H
#define CONFORMANCE_CONFORMANCE_TEST_H

#include <functional>
#include <string>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/wire_format_lite.h>

#include "third_party/jsoncpp/json.h"

namespace conformance {
class ConformanceRequest;
class ConformanceResponse;
}  // namespace conformance

namespace protobuf_test_messages {
namespace proto3 {
class TestAllTypesProto3;
}  // namespace proto3
}  // namespace protobuf_test_messages

namespace google {
namespace protobuf {

class ConformanceTestRunner {
 public:
  virtual ~ConformanceTestRunner() {}

  // Call to run a single conformance test.
  //
  // "input" is a serialized conformance.ConformanceRequest.
  // "output" should be set to a serialized conformance.ConformanceResponse.
  //
  // If there is any error in running the test itself, set "runtime_error" in
  // the response.
  virtual void RunTest(const std::string& test_name,
                       const std::string& input,
                       std::string* output) = 0;
};

// Class representing the test suite itself.  To run it, implement your own
// class derived from ConformanceTestRunner and then write code like:
//
//    class MyConformanceTestRunner : public ConformanceTestRunner {
//     public:
//      virtual void RunTest(...) {
//        // INSERT YOUR FRAMEWORK-SPECIFIC CODE HERE.
//      }
//    };
//
//    int main() {
//      MyConformanceTestRunner runner;
//      google::protobuf::ConformanceTestSuite suite;
//
//      std::string output;
//      suite.RunSuite(&runner, &output);
//    }
//
class ConformanceTestSuite {
 public:
  ConformanceTestSuite() : verbose_(false), enforce_recommended_(false) {}

  void SetVerbose(bool verbose) { verbose_ = verbose; }

  // Sets the list of tests that are expected to fail when RunSuite() is called.
  // RunSuite() will fail unless the set of failing tests is exactly the same
  // as this list.
  //
  // The filename here is *only* used to create/format useful error messages for
  // how to update the failure list.  We do NOT read this file at all.
  void SetFailureList(const std::string& filename,
                      const std::vector<std::string>& failure_list);

  // Whether to require the testee to pass RECOMMENDED tests. By default failing
  // a RECOMMENDED test case will not fail the entire suite but will only
  // generated a warning. If this flag is set to true, RECOMMENDED tests will
  // be treated the same way as REQUIRED tests and failing a RECOMMENDED test
  // case will cause the entire test suite to fail as well. An implementation
  // can enable this if it wants to be strictly conforming to protobuf spec.
  // See the comments about ConformanceLevel below to learn more about the
  // difference between REQUIRED and RECOMMENDED test cases.
  void SetEnforceRecommended(bool value) {
    enforce_recommended_ = value;
  }

  // Run all the conformance tests against the given test runner.
  // Test output will be stored in "output".
  //
  // Returns true if the set of failing tests was exactly the same as the
  // failure list.  If SetFailureList() was not called, returns true if all
  // tests passed.
  bool RunSuite(ConformanceTestRunner* runner, std::string* output);

 private:
  // Test cases are classified into a few categories:
  //   REQUIRED: the test case must be passed for an implementation to be
  //             interoperable with other implementations. For example, a
  //             parser implementaiton must accept both packed and unpacked
  //             form of repeated primitive fields.
  //   RECOMMENDED: the test case is not required for the implementation to
  //                be interoperable with other implementations, but is
  //                recommended for best performance and compatibility. For
  //                example, a proto3 serializer should serialize repeated
  //                primitive fields in packed form, but an implementation
  //                failing to do so will still be able to communicate with
  //                other implementations.
  enum ConformanceLevel {
    REQUIRED = 0,
    RECOMMENDED = 1,
  };
  string ConformanceLevelToString(ConformanceLevel level);

  void ReportSuccess(const std::string& test_name);
  void ReportFailure(const string& test_name,
                     ConformanceLevel level,
                     const conformance::ConformanceRequest& request,
                     const conformance::ConformanceResponse& response,
                     const char* fmt, ...);
  void ReportSkip(const string& test_name,
                  const conformance::ConformanceRequest& request,
                  const conformance::ConformanceResponse& response);
  void RunTest(const std::string& test_name,
               const conformance::ConformanceRequest& request,
               conformance::ConformanceResponse* response);
  void RunValidInputTest(const string& test_name,
                         ConformanceLevel level,
                         const string& input,
                         conformance::WireFormat input_format,
                         const string& equivalent_text_format,
                         conformance::WireFormat requested_output,
                         bool isProto3);
  void RunValidBinaryInputTest(const string& test_name,
                               ConformanceLevel level,
                               const string& input,
                               conformance::WireFormat input_format,
                               const string& equivalent_wire_format,
                               conformance::WireFormat requested_output,
                               bool isProto3);
  void RunValidJsonTest(const string& test_name,
                        ConformanceLevel level,
                        const string& input_json,
                        const string& equivalent_text_format);
  void RunValidJsonTestWithProtobufInput(
      const string& test_name,
      ConformanceLevel level,
      const protobuf_test_messages::proto3::TestAllTypesProto3& input,
      const string& equivalent_text_format);
  void RunValidProtobufTest(const string& test_name, ConformanceLevel level,
                            const string& input_protobuf,
                            const string& equivalent_text_format,
                            bool isProto3);
  void RunValidBinaryProtobufTest(const string& test_name,
                                  ConformanceLevel level,
                                  const string& input_protobuf,
                                  bool isProto3);
  void RunValidProtobufTestWithMessage(
      const string& test_name, ConformanceLevel level,
      const Message *input,
      const string& equivalent_text_format,
      bool isProto3);

  typedef std::function<bool(const Json::Value&)> Validator;
  void RunValidJsonTestWithValidator(const string& test_name,
                                     ConformanceLevel level,
                                     const string& input_json,
                                     const Validator& validator);
  void ExpectParseFailureForJson(const string& test_name,
                                 ConformanceLevel level,
                                 const string& input_json);
  void ExpectSerializeFailureForJson(const string& test_name,
                                     ConformanceLevel level,
                                     const string& text_format);
  void ExpectParseFailureForProtoWithProtoVersion (const string& proto,
                                                   const string& test_name,
                                                   ConformanceLevel level,
                                                   bool isProto3);
  void ExpectParseFailureForProto(const std::string& proto,
                                  const std::string& test_name,
                                  ConformanceLevel level);
  void ExpectHardParseFailureForProto(const std::string& proto,
                                      const std::string& test_name,
                                      ConformanceLevel level);
  void TestPrematureEOFForType(google::protobuf::FieldDescriptor::Type type);
  void TestIllegalTags();
  template <class MessageType>
  void TestOneofMessage (MessageType &message,
                         bool isProto3);
  template <class MessageType>
  void TestUnknownMessage (MessageType &message,
                           bool isProto3);
  void TestValidDataForType(
      google::protobuf::FieldDescriptor::Type,
      std::vector<std::pair<std::string, std::string>> values);
  bool CheckSetEmpty(const std::set<string>& set_to_check,
                     const std::string& write_to_file, const std::string& msg);
  ConformanceTestRunner* runner_;
  int successes_;
  int expected_failures_;
  bool verbose_;
  bool enforce_recommended_;
  std::string output_;
  std::string failure_list_filename_;

  // The set of test names that are expected to fail in this run, but haven't
  // failed yet.
  std::set<std::string> expected_to_fail_;

  // The set of test names that have been run.  Used to ensure that there are no
  // duplicate names in the suite.
  std::set<std::string> test_names_;

  // The set of tests that failed, but weren't expected to.
  std::set<std::string> unexpected_failing_tests_;

  // The set of tests that succeeded, but weren't expected to.
  std::set<std::string> unexpected_succeeding_tests_;

  // The set of tests that the testee opted out of;
  std::set<std::string> skipped_;

  std::unique_ptr<google::protobuf::util::TypeResolver> type_resolver_;
  std::string type_url_;
};

}  // namespace protobuf
}  // namespace google

#endif  // CONFORMANCE_CONFORMANCE_TEST_H
