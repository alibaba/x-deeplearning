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

#include "xdl/core/utils/file_utils.h"

#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

#include "xdl/core/utils/string_utils.h"

using namespace std;

namespace xdl {

bool FileUtils::DeleteLocalFile(const std::string &localFilePath) {
  return (0 == unlink(localFilePath.c_str()));
}

string FileUtils::ReadLocalFile(const std::string &filePath) {
  ifstream in(filePath.c_str());
  stringstream ss;
  string line;
  if (!in) {
    return string("");
  }
  while (getline(in, line)) {
    ss << line;
  }
  in.close();
  return ss.str();
}

string FileUtils::ReadLocalBinaryFile(const std::string &filePath) {
  struct stat stat;  
  if (lstat(filePath.c_str(), &stat) != 0) {
    return "";
  }

  size_t len = stat.st_size;
  if (len == 0) return "";
  ifstream in(filePath.c_str(), std::ios::binary);  
  if (!in) return "";
  string result;
  result.resize(len);
  in.read(&result[0], len);
  return result;
}

bool FileUtils::WriteLocalFile(const string &filePath, 
                               const string &content) {
  std::ofstream file(filePath.c_str());
  if (!file) {
    return false;
  }
  file.write(content.c_str(), content.length());
  file.flush();
  file.close();
  return true;
}

bool FileUtils::CompFile(const string& first, 
                         const string& second) {
  return (CompFile(first.c_str(), second.c_str()) == 0);
}

int FileUtils::CompFile(const char* first, 
                        const char* second) {
  if (first == NULL || second == NULL) {
    return -1;
  }

  struct stat fStat, sStat;

  int ret;
  ret = lstat(first, &fStat);
  if (ret < 0) {
    return -1;
  }

  ret = lstat(second, &sStat);
  if (ret < 0) {
    return -1;
  }

  if (fStat.st_size != sStat.st_size) {
    return 1;
  }

  FILE *ff, *sf;
  ff = fopen(first, "rb");
  if (ff == NULL) {
    return -1;
  }

  sf = fopen(second, "rb");
  if (sf == NULL) {
    fclose(ff);
    return -1;
  }

  char fBuf[1024], sBuf[1024];
  int fc, result = 0;
  while (1) {
    fc = fread(fBuf, 1, 1024, ff);
    if (fc == 0) {
      break;
    }

    fread(sBuf, 1, 1024, sf);
    ret = memcmp(fBuf, sBuf, fc);
    if (ret != 0) {
      result = 1;
      break;
    }
  }

  fclose(sf);
  fclose(ff);
  return result;
}

bool FileUtils::IsDirExists(const std::string& dirPath) {
  struct stat st;
  if (stat(dirPath.c_str(), &st) != 0) {
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    return false;
  }
  return true;
}

bool FileUtils::MoveFile(const std::string& srcFile,
                         const std::string& desFile) {
  std::string cmd =  "mv -f " + srcFile + " " + desFile;

  int ret = system(cmd.c_str());

  return (ret == 0);
}

bool FileUtils::TouchFile(const std::string& file) {
  return WriteLocalFile(file, "");
}

bool FileUtils::CreatDir(const std::string& dir, mode_t mode) {
  if (dir.empty()) {
    return false;
  }

  string tmpDir = dir;
  if (tmpDir[tmpDir.size()-1] == '/') {
    tmpDir = tmpDir.substr(0, tmpDir.size() - 1);
  }
    
  vector<string> paths = StringUtils::split(tmpDir, "/", false);
  string path;
  for (size_t i = 0; i< paths.size(); ++i) {
    path += paths[i] + "/";
    if (!IsDirExists(path) &&
        mkdir(path.c_str(), mode) != 0)
    {
      return false;
    }
  }

  return true;
}

bool FileUtils::CopyFile(const std::string& srcFile,
                         const std::string& desFile) {
  std::string cmd =  "cp -f " + srcFile + " " + desFile;

  int ret = system(cmd.c_str());

  return (ret == 0);
}

bool FileUtils::IsFileExist(const string& filePath) {
  return access(filePath.c_str(), F_OK) == 0;
}

size_t FileUtils::FileSize(const std::string& path) {
  if (!IsFileExist(path)) {
    return 0;
  }

  struct stat buf;
  int32_t ret;
  if ((ret = lstat(path.c_str(), &buf)) != 0) {
    return 0;
  }

  return buf.st_size;
}  // FileSize

} //ps
