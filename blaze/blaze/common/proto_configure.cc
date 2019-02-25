/*
 * \file proto_configure.cc
 * \desc The protobuf config parser
 */
#include "blaze/common/proto_configure.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include <fstream>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

using namespace google::protobuf;
using namespace std;

namespace {

const char* kErrors[] = {
  "",
  "No Such Proto Type",
  "Generate Proto Type Failed",
  "Open Config File Failed",
  "Parse Proto Message Failed",
};

}  // namespace

namespace blaze {

const char* ProtoConfigure::ErrorMessage(ProtoConfigure::Status status) {
  return kErrors[status];
}

ProtoConfigure::Status
ProtoConfigure::Init(const string& type_name, const string& conf_name) {
  // create protobuf message by class name
  const Descriptor* descriptor =
      DescriptorPool::generated_pool()->FindMessageTypeByName(type_name);
  if (!descriptor) return kNoSuchProtoType;
  const Message* prototype =
      MessageFactory::generated_factory()->GetPrototype(descriptor);
  if (!prototype) return kFailGenProtoType;
  Message* message = prototype->New();

  int fd = open(conf_name.c_str(), O_RDONLY);
  if (fd < 0) return kFailOpenConfigFile;

  io::FileInputStream file(fd);
  // By default, the file descriptor is not closed when the stream is
  // destroyed.  Call SetCloseOnDelete(true) to change that.
  file.SetCloseOnDelete(true);
  if (!TextFormat::Parse(&file, message)) return kFailParseProtoMessage;

  Release();
  config_ = message;
  type_name_ = type_name;
  conf_name_ = conf_name;

  return kOK;
}

ProtoConfigure::Status
ProtoConfigure::InitByTextConf(const std::string &type_name, const std::string &text_conf) {
  // create protobuf message by class name
  const Descriptor* descriptor =
      DescriptorPool::generated_pool()->FindMessageTypeByName(type_name);
  if (!descriptor) return kNoSuchProtoType;
  const Message* prototype =
      MessageFactory::generated_factory()->GetPrototype(descriptor);
  if (!prototype) return kFailGenProtoType;
  Message* message = prototype->New();

  if (!TextFormat::ParseFromString(text_conf, message)) return kFailParseProtoMessage;

  Release();
  config_ = message;
  type_name_ = type_name;
  conf_name_ = "";

  return kOK;
}

void ProtoConfigure::Release() {
  if (config_ != NULL) {
    delete config_;
    config_ = NULL;
  }
}

ProtoConfigure::Status
ProtoConfigure::GenProtoConfigFile(const Message& message,
                                   const string& output_file) {
  ofstream fout;
  fout.open(output_file.c_str(), ios::out | ios_base::ate);
  if (!fout.is_open()) return kFailOpenConfigFile;

  string text;
  TextFormat::PrintToString(message, &text);

  fout << text << endl;
  fout.flush();
  fout.close();

  return kOK;
}

}  // namespace blaze
