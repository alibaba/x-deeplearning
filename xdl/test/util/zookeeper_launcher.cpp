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

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <sstream>
#include <iostream>
#include <string>

#include "zookeeper_launcher.h"
#include "zookeeper_define.h"

using namespace std;

namespace ps {

std::string ZookeeperLauncher::Exec(const char* cmd) {
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

void ZookeeperLauncher::Start(void) {
  stringstream ss;
  ss << "cd " << XDL_ZOOKEEPER_PATH << "; bin/zkServer.sh start";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
}

void ZookeeperLauncher::Stop(void) {
  stringstream ss;
  ss << "cd " << XDL_ZOOKEEPER_PATH << "; bin/zkServer.sh stop; rm -rf zookeeper.out";
  string out = Exec(ss.str().c_str());
  cout << out << std::endl;
  out = Exec("rm -rf /tmp/xdl_zookeeper");
  cout << out << std::endl;
}

}
