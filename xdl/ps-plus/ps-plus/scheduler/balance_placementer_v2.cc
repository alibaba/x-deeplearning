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

#include "ps-plus/scheduler/placementer.h"
#include "ps-plus/common/hasher.h"
#include "ps-plus/common/logging.h"

#include <map>
#include <vector>
#include <algorithm>

namespace ps {
namespace scheduler {

static const double kHashMem = 128;
static const double kDefaultMemRatio = 2;

namespace {

struct VariableInfos {
  struct Variable {
    enum Type {
      kDense,
      kSparse,
      kHash
    };
    Type type;
    std::string name;
    double slice_mem;
    double slice_net;
    size_t slice_num;
    bool no_split;
  };

  struct ServerInfo {
    double mem;
    double net;
    size_t socket;
  };

  std::vector<Variable> dense_vars;
  std::vector<Variable> sparse_vars;
  std::vector<Variable> hash_vars;
  std::vector<ServerInfo> servers;
  size_t mem;
  double avg_net;
  double avg_sparse_mem;
};

struct Solution {
  struct Part {
    size_t server;
    size_t size;
  };
  struct Server {
    double mem;
    double net;
    size_t socket;
  };
  std::unordered_map<std::string, std::vector<Part>> parts_map;
  std::vector<Server> servers;
};

bool CreateBase(const VariableInfos& infos, Solution* solution) {
  for (auto server : infos.servers) {
    solution->servers.push_back(Solution::Server{.mem = server.mem, .net = server.net, .socket = server.socket});
  }
  std::vector<VariableInfos::Variable> vars;
  vars.insert(vars.end(), infos.dense_vars.begin(), infos.dense_vars.end());
  vars.insert(vars.end(), infos.sparse_vars.begin(), infos.sparse_vars.end());
  vars.insert(vars.end(), infos.hash_vars.begin(), infos.hash_vars.end());
  //solution->parts.resize(vars.size());
  size_t s = 0;
  for (size_t i = 0; i < vars.size(); i++) {
    size_t a = vars[i].slice_num;
    while (a > 0 && s < solution->servers.size()) {
      size_t size = std::min((size_t)((infos.mem - solution->servers[s].mem) / vars[i].slice_mem + 1), a);
      solution->servers[s].mem += vars[i].slice_mem * size;
      //solution->parts[i].push_back(Solution::Part{.server = s, .size = size});
      a -= size;
      if (solution->servers[s].mem >= infos.mem) {
        s++;
      }
    }
    if (s >= solution->servers.size()) {
      break;
    }
  }
  if (s >= solution->servers.size()) {
    return false;
  }
  return true;
}

void CreateBalance(const VariableInfos& infos, Solution* solution) {
  for (auto server : infos.servers) {
    solution->servers.push_back(Solution::Server{.mem = server.mem, .net = server.net, .socket = server.socket});
  }
  size_t s = 0;
  size_t total_socket = infos.dense_vars.size() + infos.sparse_vars.size() + infos.hash_vars.size();
  double avg_net = infos.avg_net;
  double avg_sparse_mem = infos.avg_sparse_mem;
  for (size_t i = 0; i < infos.hash_vars.size(); ++i) {
    size_t a = infos.hash_vars[i].slice_num;
    solution->parts_map[infos.hash_vars[i].name] = std::vector<Solution::Part>();
    for (size_t j = 0; j < solution->servers.size(); ++j) {
      size_t size = std::min((size_t)(Hasher::kTargetRange / solution->servers.size()) + 1, a);
      solution->servers[j].mem += infos.hash_vars[i].slice_mem * size;
      solution->servers[j].socket++;
      if (a != size) { total_socket++; }
      solution->parts_map[infos.hash_vars[i].name].push_back(Solution::Part{.server = j, .size = size});
      a -= size;
      if (a == 0) { break; }
    }
  }
  s = 0;
  for (size_t i = 0; i < infos.sparse_vars.size(); ++i) {
    size_t a = infos.sparse_vars[i].slice_num;
    solution->parts_map[infos.sparse_vars[i].name] = std::vector<Solution::Part>();
    while (a > 0 && s < solution->servers.size()) {
      size_t size;
      if (infos.avg_sparse_mem < solution->servers[s].mem) {
        s++;
        continue;
      };
      size = std::min((size_t)((infos.avg_sparse_mem - solution->servers[s].mem) / infos.sparse_vars[i].slice_mem + 1) , a);
      solution->servers[s].mem += infos.sparse_vars[i].slice_mem * size;
      solution->servers[s].socket++;
      if (a != size) { total_socket++; }
      solution->parts_map[infos.sparse_vars[i].name].push_back(Solution::Part{.server = s, .size = size});
      a -= size;
      if (solution->servers[s].mem >= infos.avg_sparse_mem) {
        s++;
      }
    }
    if (s >= solution->servers.size()) {
      break;
    }
  }
  if (s >= solution->servers.size()) {
    LOG(INFO) << "Sparse variable memory out of range";
  }
  s = 0;
  for (size_t i = 0; i < infos.dense_vars.size(); ++i) {
    size_t a = infos.dense_vars[i].slice_num;
    solution->parts_map[infos.dense_vars[i].name] = std::vector<Solution::Part>();
    while (a > 0 && s < solution->servers.size()) {
      size_t size;
      if (infos.dense_vars[i].slice_net == 0) {
        size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.dense_vars[i].slice_mem) , a);
        if ((double)(size * solution->servers.size()) / a < 0.2) {
          s++;
          if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
          continue;
        }
      } else {
        size = std::min((size_t)((avg_net - solution->servers[s].net) / infos.dense_vars[i].slice_net) , a);
        size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.dense_vars[i].slice_mem) , size);
        if ((double)(size * solution->servers.size()) / a < 0.2) {
          s++;
          if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
          continue;
        }
      }
      if (infos.dense_vars[i].no_split == true && size != a) {
        s++;
        if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
        continue;
      }
      solution->servers[s].mem += infos.dense_vars[i].slice_mem * size;
      solution->servers[s].net += infos.dense_vars[i].slice_net * size;
      if (infos.dense_vars[i].slice_net != 0) { solution->servers[s].socket++; }
      if (a != size) { total_socket++; }
      size_t iter;
      for (iter = 0; iter < solution->parts_map[infos.dense_vars[i].name].size(); ++iter) {
        if (s == solution->parts_map[infos.dense_vars[i].name][iter].server) {
          solution->parts_map[infos.dense_vars[i].name][iter].size += size;
          break;
        }
      }
      if (iter == solution->parts_map[infos.dense_vars[i].name].size()) {
        solution->parts_map[infos.dense_vars[i].name].push_back(Solution::Part{.server = s, .size = size});
      }
      a -= size;
      if (solution->servers[s].mem >= infos.mem || solution->servers[s].net >= avg_net 
        || solution->servers[s].socket >= total_socket / solution->servers.size() + 2) {
        s++;
      }
      if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
    }
  }     
}

bool Get(const VariableInfos& infos, Solution* solution) {
  Solution sol;
  if (!CreateBase(infos, &sol)) {
    return false;
  }
  CreateBalance(infos, solution);
  return true;
}

}

class BalancePlacementerV2 : public Placementer {
 public:
  virtual Status Placement(const std::vector<VariableInfo>& inputs, std::vector<VariableInfo>* outputs, const Arg& arg, size_t server) override {
    VariableInfos infos;
    infos.mem = arg.mem;
    double total_net = 0;
    double total_sparse_mem = 0;
    for (size_t i = 0; i < server; ++i) {
      infos.servers.push_back(VariableInfos::ServerInfo{.mem = 0, .net = 0, .socket = 0});
    }
    for (const VariableInfo& info : inputs) {
      VariableInfos::Variable x;
      x.name = info.name;
      auto argiter = info.args.find("mem_ratio");
      double mem_ratio = argiter == info.args.end() ? kDefaultMemRatio : atof(argiter->second.c_str());
      argiter = info.args.find("no_split");
      x.no_split = argiter == info.args.end() ? false : true;
      if (info.type == VariableInfo::kIndex) {
        if (info.shape.empty()) {
          x.slice_num = 1;
          x.slice_mem = SizeOfType(info.datatype) * mem_ratio;
          x.slice_net = SizeOfType(info.datatype);
          x.type = VariableInfos::Variable::kDense;
        } else {
          size_t slice_size = 1;
          for (size_t i = 1; i < info.shape.size(); i++) {
            slice_size *= info.shape[i];
          }
          auto iter = info.args.find("batch_read");
          double ratio = iter == info.args.end() ? 1 : atof(iter->second.c_str()) / info.shape[0];
          x.type = iter == info.args.end() ? VariableInfos::Variable::kDense : VariableInfos::Variable::kSparse;
          x.slice_num = info.shape[0];
          x.slice_mem = SizeOfType(info.datatype) * slice_size * mem_ratio;
          iter = info.args.find("io_ratio");
          double io_ratio = iter == info.args.end() ? 1 : atof(iter->second.c_str());
          x.slice_net = SizeOfType(info.datatype) * slice_size * ratio * io_ratio;
          if (x.type == VariableInfos::Variable::kSparse) {
            total_sparse_mem += x.slice_num * x.slice_mem;
          }
        }
      } else if (info.type == VariableInfo::kHash128 || info.type == VariableInfo::kHash64) {
        if (info.shape.empty()) {
          return Status::ArgumentError("Hash Should at least 1 dim");
        }
        size_t slice_size = 1;
        for (size_t i = 1; i < info.shape.size(); i++) {
          slice_size *= info.shape[i];
        }
        auto iter = info.args.find("batch_read");
        int64_t batch_read = iter == info.args.end() ? info.shape[0] : atoll(iter->second.c_str());
        x.type = VariableInfos::Variable::kHash;
        x.slice_num = Hasher::kTargetRange;
        //x.slice_mem = double((SizeOfType(info.datatype) * slice_size * mem_ratio + kHashMem) * info.shape[0]) * 2 / Hasher::kTargetRange;
        x.slice_mem = 1;
        iter = info.args.find("io_ratio");
        double io_ratio = iter == info.args.end() ? 1 : atof(iter->second.c_str());
        x.slice_net = double(SizeOfType(info.datatype) * slice_size * batch_read * io_ratio) / Hasher::kTargetRange;
      } else {
        return Status::NotImplemented("Balance PlacementerV2 not support type: " + std::to_string(info.type) + " @ " + info.name);
      }
      if(x.type == VariableInfos::Variable::kDense) {
        total_net += x.slice_net * x.slice_num;
      }
      if (info.parts.empty()) {
        if (x.type == VariableInfos::Variable::kDense) { infos.dense_vars.push_back(x); }
        else if (x.type == VariableInfos::Variable::kSparse) { infos.sparse_vars.push_back(x); }
        else if (x.type == VariableInfos::Variable::kHash) { infos.hash_vars.push_back(x); }
      } else {
        for (auto&& part : info.parts) {
          if (x.type == VariableInfos::Variable::kDense) {
            infos.servers[part.server].net += x.slice_net * part.size;
          }
          infos.servers[part.server].mem += x.slice_mem * part.size;
          infos.servers[part.server].socket++;
        }
      }
    }
    infos.avg_net = total_net / server;
    infos.avg_sparse_mem = total_sparse_mem / server;
    Solution solution;
    if (infos.dense_vars.size() + infos.sparse_vars.size() + infos.hash_vars.size() > 0) {
      bool result = Get(infos, &solution);
      if (!result) {
        return Status::ArgumentError("Cannot Placement, Too Heavy");
      }
    }
    outputs->clear();
    for (VariableInfo info : inputs) {
      if (info.parts.empty()) {
        auto parts_iter = solution.parts_map.find(info.name);
        for (auto item : parts_iter->second) {
          info.parts.push_back(VariableInfo::Part{.server = item.server, .size = item.size});
        }
      }
      outputs->push_back(info);
    }
    return Status::Ok();
  }
};

PLUGIN_REGISTER(Placementer, BalanceV2, BalancePlacementerV2);

}
}

