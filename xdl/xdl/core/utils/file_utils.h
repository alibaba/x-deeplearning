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

#ifndef XDL_CORE_UTILS_FILE_UTILS_H
#define XDL_CORE_UTILS_FILE_UTILS_H

#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

namespace xdl {

class FileUtils {
 public:
 public:
  FileUtils() = default;
  ~FileUtils()=  default;

 public:
  static std::string ReadLocalFile(const std::string &filePath);
  static std::string ReadLocalBinaryFile(const std::string &filePath);

  static void ReadLocalFile(const std::string &filePath,
                            std::vector<std::string>* lines);

  static bool WriteLocalFile(const std::string &filePath,
                             const std::string &content);
  static bool DeleteLocalFile(const std::string &localFilePath);

  static bool CompFile(const std::string& first, const std::string& second);

  static int CompFile(const char* first, const char* second);

  static bool MoveFile(const std::string& srcFile,
                       const std::string& desFile);

  static bool TouchFile(const std::string& file);

  static bool CreatDir(const std::string& dir, mode_t mode = S_IRWXU);

  static bool CopyFile(const std::string& srcFile,
                       const std::string& desFile);

  static bool IsFileExist(const std::string& filePath);

  static bool IsDirExists(const std::string& dirPath);

  static size_t FileSize(const std::string& path);
};

} //xdl

#endif  // XDL_CORE_UTILS_FILE_UTILS_H
