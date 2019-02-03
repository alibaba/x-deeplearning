/*!
 * \file model_importer.cc
 * \brief base model importer
 */
#include "blaze/model_importer/model_importer.h"

#include <sys/stat.h>

#include "blaze/common/log.h"
#include "blaze/common/common_defines.h"

namespace blaze {

ModelImporter::ModelImporter() : data_type_(kFloat) {
  net_def_.set_version(BLAZE_VERSION_MAJOR * 1000 + BLAZE_VERSION_MINOR);
}

bool ModelImporter::SaveToTextFile(const char* blaze_model_file) {
  const auto& content = net_def_.DebugString();
  return SaveFileContent(blaze_model_file, content);
}

bool ModelImporter::SaveToBinaryFile(const char* blaze_model_file) {
  std::string content;
  uint32_t msg_size = static_cast<uint32_t>(net_def_.ByteSize());
  content.resize(msg_size);
  net_def_.SerializeToArray(const_cast<char*>(content.c_str()), msg_size);
  return SaveFileContent(blaze_model_file, content);
}
  
bool ModelImporter::ReadFileContent(const char* filename, std::string* content) {
  struct stat st;
  if (stat(filename, &st)) {
    LOG_ERROR("model file: %s not exist", filename);
    return false;
  }
  content->resize(st.st_size);
  FILE* fp = fopen(filename, "r");
  if (fp == nullptr) {
    LOG_ERROR("open model file: %s failed", filename);
    return false;
  }
  fread(const_cast<char*>(content->c_str()), 1, content->length(), fp);
  fclose(fp);
  return true;
}

bool ModelImporter::SaveFileContent(const char* filename, const std::string& content) {
  FILE* fp = fopen(filename, "w");
  if (fp == nullptr) {
    LOG_ERROR("open and create model file: %s failed", filename);
    return false;
  }
  fwrite(const_cast<char*>(content.c_str()), 1, content.length(), fp);
  fclose(fp);
  return true;
}

std::string ModelImporter::GetParentPath(const std::string& path) {
  if (path.empty()) {
    return ".";
  }
  size_t delim_pos = path.rfind('/', path.length() - 2);
  if (std::string::npos == delim_pos) {
    return ".";
  } else {
    return path.substr(0, delim_pos + 1);
  }
}

}  // namepace blaze
