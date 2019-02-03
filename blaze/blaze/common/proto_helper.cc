/*
 * \file proto_helper.cc 
 * \brief The proto helper functions implementation
 */
#include "blaze/common/proto_helper.h"

#include <sys/stat.h>
#include <utility>
#include <vector>

#include "blaze/common/exception.h"
#include "blaze/common/log.h"
#include "blaze/common/proto_configure.h"
#include "blaze/common/types.h"

namespace {
template <typename InputType, typename TargetType>
bool SupportsLosslessConversion(const InputType value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
}  // namespace

namespace blaze {

// Load netdef from binary file
bool NetDefHelper::LoadNetDefFromBinaryFile(const char* filename, NetDef* net_def) {
  struct stat st;
  if (stat(filename, &st)) {
    LOG_ERROR("model file: %s not exist", filename);
    return false;
  }
  std::string content;
  content.resize(st.st_size);
  FILE* fp = fopen(filename, "r");
  if (fp == nullptr) {
    LOG_ERROR("open model file: %s failed", filename);
    return false;
  }
  fread(const_cast<char*>(content.c_str()), 1, content.length(), fp);
  fclose(fp);
  bool success = net_def->ParseFromArray(content.c_str(), content.length());
  if (!success) {
    LOG_ERROR("parse NetDef from %s failed", filename);
    return false;
  }
  return true;
}

bool NetDefHelper::LoadNetDefFromTextFile(const char* filename, NetDef* net_def) {
  try {
    ProtoConfigure proto_conf("blaze.NetDef", filename);
    const NetDef* config = dynamic_cast<const NetDef*>(proto_conf.config());
    *net_def = *config;
  } catch (...) {
    LOG_ERROR("Open net_file %s failed", filename);
    return false;
  }
  return true;
}

bool NetDefHelper::SaveNetDefToBinaryFile(const char* filename, NetDef* net_def) {
  std::string content;
  uint32_t msg_size = static_cast<uint32_t>(net_def->ByteSize());
  content.resize(msg_size);
  net_def->SerializeToArray(const_cast<char*>(content.c_str()), msg_size);
  FILE* fp = fopen(filename, "w");
  if (fp == nullptr) {
    LOG_ERROR("save model file %s failed", filename);
    return false;
  }
  fwrite(const_cast<char*>(content.c_str()), 1, content.length(), fp);
  fclose(fp);
  return true;
}

bool NetDefHelper::SaveNetDefToTextFile(const char* filename, NetDef* net_def) {
  FILE* fp = fopen(filename, "w");
  if (fp == nullptr) {
    LOG_ERROR("save model file %s failed", filename);
    return false;
  }
  const std::string debug_string = net_def->DebugString();
  fwrite(const_cast<char*>(debug_string.c_str()), 1, debug_string.length(), fp);
  fclose(fp);
  return true;
}

// ArgumentHelper implementation
ArgumentHelper::ArgumentHelper(const OperatorDef& def) {
  for (auto& arg : def.arg()) {
    InsertArgument(arg);
  }
}

ArgumentHelper::ArgumentHelper(const NetDef& def) {
  for (auto& arg : def.arg()) {
   InsertArgument(arg);
  }
}

void ArgumentHelper::InsertArgument(const Argument& arg) {
  if (arg_map_.count(arg.name())) {
    LOG_INFO("argument name=%s", arg.name().c_str());
    if (arg.SerializeAsString() != arg_map_[arg.name()].SerializeAsString()) {
      BLAZE_THROW("Found argument of the same name ", arg.name());
    } else {
      LOG_ERROR("Duplicated argument name %s", arg.name().c_str());
    }
  }
  arg_map_[arg.name()] = arg;
}

bool ArgumentHelper::HasArgument(const std::string& name) const {
  return arg_map_.count(name);
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname, enforce_lossless_conversion)      \
  template <>                                                                           \
  T ArgumentHelper::GetSingleArgument<T>(const std::string& name,                       \
                                         const T& default_value) const {                \
    if (arg_map_.count(name) == 0) {                                                    \
      return default_value;                                                             \
    }                                                                                   \
    CHECK(arg_map_.at(name).has_##fieldname(), "%s filed missing", #fieldname);          \
    auto value = arg_map_.at(name).fieldname();                                         \
    if (enforce_lossless_conversion) {                                                  \
      auto supportsConversion = SupportsLosslessConversion<decltype(value), T>(value);  \
      CHECK(supportsConversion, "should support lossless conversion");                  \
    }                                                                                   \
    return static_cast<T>(value);                                                       \
  }                                                                                     \
  template <>                                                                           \
  bool ArgumentHelper::HasSingleArgumentOfType<T>(const std::string& name) const {      \
    if (arg_map_.count(name) == 0) {                                                    \
      return false;                                                                     \
    }                                                                                   \
    return arg_map_.at(name).has_##fieldname();                                         \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float16, f, false);
INSTANTIATE_GET_SINGLE_ARGUMENT(float, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(double, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(int8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(size_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(std::string, s, false);

#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname, enforce_lossless_conversion)    \
  template <>                                                                           \
  std::vector<T> ArgumentHelper::GetRepeatedArgument<T>(const std::string& name,        \
                                      const std::vector<T>& default_value) const {      \
    if (arg_map_.count(name) == 0) {                                                    \
      return default_value;                                                             \
    }                                                                                   \
    std::vector<T> values;                                                              \
    for (const auto& v : arg_map_.at(name).fieldname()) {                               \
      if (enforce_lossless_conversion) {                                                \
        auto supportsConversion =                                                       \
          SupportsLosslessConversion<decltype(v), T>(v);                                \
        CHECK(supportsConversion, "should support lossless conversion");                \
      }                                                                                 \
      values.push_back(static_cast<T>(v));                                              \
    }                                                                                   \
    return values;                                                                      \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float16, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(double, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(int8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(size_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(std::string, strings, false)

#undef INSTANTIATE_GET_REPEATED_ARGUMENT

#define INSTANTIATE_SET_SINGLE_ARGUMENT(T, fieldname)                                 \
   template <>                                                                        \
   void ArgumentHelper::SetSingleArgument(OperatorDef& op, const std::string& name,   \
                                          const T& value) {                           \
     for (auto& arg : *(op.mutable_arg())) {                                          \
       if (arg.name() == name) {                                                      \
          arg.set_##fieldname(value);                                                 \
          return;                                                                     \
       }                                                                              \
     }                                                                                \
     auto arg = op.add_arg();                                                         \
     arg->set_name(name);                                                             \
     arg->set_##fieldname(value);                                                     \
   }

INSTANTIATE_SET_SINGLE_ARGUMENT(float16, f)
INSTANTIATE_SET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_SET_SINGLE_ARGUMENT(double, f)
INSTANTIATE_SET_SINGLE_ARGUMENT(bool, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(int8_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(int16_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(int, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(int64_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(uint8_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(uint16_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(size_t, i)
INSTANTIATE_SET_SINGLE_ARGUMENT(std::string, s);

#undef INSTANTIATE_SET_SINGLE_ARGUMENT

#define INSTANTIATE_SET_REPEATED_ARGUMENT(T, fieldname)                               \
   template <>                                                                        \
   void ArgumentHelper::SetRepeatedArgument(OperatorDef& op, const std::string& name, \
                                            const std::vector<T>& values) {           \
     for (auto& arg : *(op.mutable_arg())) {                                          \
       if (arg.name() == name) {                                                      \
         arg.clear_##fieldname();                                                     \
         for (const auto& value : values) {                                           \
           arg.add_##fieldname(value);                                                \
         }                                                                            \
         return;                                                                      \
       }                                                                              \
     }                                                                                \
     auto arg = op.add_arg();                                                         \
     arg->set_name(name);                                                             \
     for (const auto& value : values) {                                               \
       arg->add_##fieldname(value);                                                   \
     }                                                                                \
   }

INSTANTIATE_SET_REPEATED_ARGUMENT(float16, floats)
INSTANTIATE_SET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_SET_REPEATED_ARGUMENT(double, floats)
INSTANTIATE_SET_REPEATED_ARGUMENT(bool, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(int8_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(int16_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(int64_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(uint8_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(uint16_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(size_t, ints)
INSTANTIATE_SET_REPEATED_ARGUMENT(std::string, strings)

#undef INSTANTIATE_SET_REPEATED_ARGUMENT

bool ArgumentHelper::ConstantValueInRange(float min, float max) const {
  std::vector<float> value = this->GetRepeatedArgument<float>("value");
  for (const auto v : value) {
    if (v < min || v > max) {
      return false;
    }
  }
  return true;
}

}  // namespace blaze

