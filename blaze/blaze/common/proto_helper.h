/*
 * \file proto_helper.h 
 * \brief The proto helper functions.
 */
#pragma once

#include <vector>

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

// load net def from binary file
class NetDefHelper {
 public:
  // load binary netdef from file
  static bool LoadNetDefFromBinaryFile(const char* filename, NetDef* net_def);
  // load text netdef from file
  static bool LoadNetDefFromTextFile(const char* filename, NetDef* net_def);
  // save netdef as binary format file
  static bool SaveNetDefToBinaryFile(const char* filename, NetDef* net_def);
  // save netdef as text format file
  static bool SaveNetDefToTextFile(const char* filename, NetDef* net_def);
};

class ArgumentHelper {
 public:
  template <typename Def>
  static bool HasArgument(const Def& def, const std::string& name) {
    return ArgumentHelper(def).HasArgument(name);
  }

  template <typename Def, typename T>
  static T GetSingleArgument(const Def& def, const std::string& name, const T& default_value) {
    return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
  }

  template <typename Def, typename T>
  static bool HasSingleArgumentOfType(const Def& def, const std::string& name) {
    return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
  }

  template <typename Def, typename T>
  static std::vector<T> GetRepeatedArgument(const Def& def, const std::string& name,
                                            const std::vector<T>& default_value = std::vector<T>()) {
    return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
  }

  template <typename Def, typename MessageType>
  static MessageType GetMessageArgument(const Def& def, const std::string& name) {
    return ArgumentHelper(def).GetMessageArgument<MessageType>(name);
  }

  template <typename Def, typename MessageType>
  static std::vector<MessageType> GetRepeatedMessageArgument(const Def& def, const std::string& name) {
    return ArgumentHelper(def).GetRepeatedMessageArgument<MessageType>(name);
  }

  // Construction method
  explicit ArgumentHelper(const OperatorDef& def);
  explicit ArgumentHelper(const NetDef& def);

  bool HasArgument(const std::string& name) const;
  template <typename T>
  T GetSingleArgument(const std::string& name, const T& default_value) const;
  template <typename T>
  bool HasSingleArgumentOfType(const std::string& name) const;

  template <typename T>
  std::vector<T> GetRepeatedArgument(const std::string& name,
                                     const std::vector<T>& default_value = std::vector<T>()) const;
  template <typename MessageType>
  MessageType GetMessageArgument(const std::string& name) const {
    CHECK(arg_map_.count(name), "cannot find parameter named %s", name.c_str());
    MessageType message;
    if (arg_map_.at(name).has_s()) {
      CHECK(message.ParseFromString(arg_map_.at(name).s()),
            "protobuf parse failed");
    }
    return message;
  }
  template <typename MessageType>
  std::vector<MessageType> GetRepeatedMessageArgument(const std::string& name) const {
    CHECK(arg_map_.count(name), "cannot find parameter named %s", name.c_str());
    std::vector<MessageType> messages(arg_map_.at(name).strings_size());
    for (int i = 0; i < messages.size(); ++i) {
      CHECK(messages[i].ParseFromString(arg_map_.at(name).strings(i)),
            "protobuf parse failed");
    }
    return messages;
  }

  // The constant op's value is in Range [min, max]
  bool ConstantValueInRange(float min, float max) const;

  // Argument in OperatorDef modification.
  static void ClearArgument(OperatorDef& op) {
    op.clear_arg();
  }
  template <typename T>
  static void SetSingleArgument(OperatorDef& op, const std::string& name, const T& value);
  template <typename T>
  static void SetRepeatedArgument(OperatorDef& op, const std::string& name, const std::vector<T>& value);
  template <typename T>
  static void SetRepeatedArgument(OperatorDef& op, const std::string& name, const T* data, size_t n) {
    std::vector<T> value;
    for (size_t i = 0; i < n; ++i) value.push_back(data[i]);
    SetRepeatedArgument<T>(op, name, value);
  }

 private:
  void InsertArgument(const Argument& arg);

  BlazeMap<std::string, Argument> arg_map_;
};

}  // namespace blaze

