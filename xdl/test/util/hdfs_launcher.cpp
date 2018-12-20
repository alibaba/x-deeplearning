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
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <sstream>
#include <iostream>
#include <string>

#include "hdfs_launcher.h"
#include "hdfs_define.h"

using namespace std;

namespace ps {

std::string HDFSLauncher::Exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe) throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
      result += buffer.data();
  }
  return result;
}

bool HDFSLauncher::Start(void) {
  /* Run hdfs test case need four environment variable:
   * JAVA_HOME, HADOOP_HDFS_HOME, CLASSPATH, LD_LIBRARY_PATH */
  char *hdfs_home = getenv("HADOOP_HDFS_HOME");
  char *java_home = getenv("JAVA_HOME");
  char *classpath = getenv("CLASSPATH");
  char *ld_path = getenv("LD_LIBRARY_PATH");

  if (hdfs_home == NULL || java_home == NULL || classpath == NULL || ld_path == NULL) {
    return false;
  }

  stringstream ss;
  ss << "cd " << XDL_HADOOP_PATH << "; sh run.sh";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
  return true;
}

void HDFSLauncher::Stop(void) {
  stringstream ss;
  ss << "cd " << XDL_HADOOP_PATH << "; sh stop.sh";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
}

}

