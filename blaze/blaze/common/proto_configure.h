/*
 * \file proto_configure.h
 * \desc The protobuf config parser
 */
#pragma once

#include <string>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace blaze {

class ProtoConfigure {
 public:
  ProtoConfigure() : type_name_(), conf_name_(), config_(nullptr) { }
  ProtoConfigure(const std::string& type_name, const std::string& conf_name)
    : type_name_(), conf_name_(), config_(nullptr) { Init(type_name, conf_name); }
  virtual ~ProtoConfigure() { Release(); }

  // status code of the ProtoConfigure operation
  enum Status {
    kOK                    = 0,
    kNoSuchProtoType       = 1,
    kFailGenProtoType      = 2,
    kFailOpenConfigFile    = 3,
    kFailParseProtoMessage = 4,
  };

  // init the protobuf configure from file:
  //       type_name is the pb conf class type name
  //       conf_name is the pb conf file path name
  // NOTE: we can do this because we have all the known proto compiled
  Status Init(const std::string& type_name, const std::string& conf_name);

  Status InitByTextConf(const std::string& type_name, const std::string& text_conf);

  // generate protobuf config file by the proto define message
  static Status GenProtoConfigFile(const google::protobuf::Message& message,
                                   const std::string& output_file);

  // show the error message by status code
  static const char* ErrorMessage(Status status);

  const std::string& conf_name() const { return conf_name_; }
  const std::string& type_name() const { return type_name_; }
  const google::protobuf::Message* config() const { return config_; }

 private:
  // disable copy and assign
  ProtoConfigure(const ProtoConfigure&);
  ProtoConfigure& operator=(const ProtoConfigure&);

  // release the config message
  void Release();

  std::string type_name_;
  std::string conf_name_;
  google::protobuf::Message* config_;
};

}  // namespace blaze
