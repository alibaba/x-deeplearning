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
#include "ps-plus/common/file_system.h"

#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <cstdlib>
#include "ps-plus/common/logging.h"

namespace ps {
namespace scheduler {

static const double kHashMem = 128;
static const double kDefaultMemRatio = 2;
static const double kSparseCpuRatio = 1.5;
static const double kHashCpuRatio = 2.0;

namespace {

struct VariableInfos {
  struct Variable {
    double slice_mem;
    double slice_net;
    double slice_cpu;
    size_t slice_num;
    std::string name;
    size_t visit_time;
    int64_t dense_visit_ids;
    int64_t sparse_visit_ids;
    bool no_split;
    int64_t dimension;
  };

  struct ServerInfo {
    double mem;
    double net;
    double cpu;
    size_t socket;
  };

  std::vector<Variable> vars;
  std::vector<ServerInfo> servers;
  size_t mem;
  double avg_net;
  double avg_cpu;
};

struct Solution {
  struct Part {
    size_t server;
    size_t size;
  };
  struct Server {
    double mem;
    double net;
    double cpu;
    size_t socket;
  };
  std::vector<std::vector<Part>> parts;
  std::vector<Server> servers;
};

bool CreateBase(const VariableInfos& infos, Solution* solution) {
  for (auto server : infos.servers) {
    solution->servers.push_back(Solution::Server{.mem = server.mem, .net = server.net, .cpu = server.cpu, .socket = server.socket});
  }
  solution->parts.resize(infos.vars.size());
  size_t s = 0;
  for (size_t i = 0; i < infos.vars.size(); i++) {
    size_t a = infos.vars[i].slice_num;
    while (a > 0 && s < solution->servers.size()) {
      size_t size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.vars[i].slice_mem + 1), a);
      solution->servers[s].mem += infos.vars[i].slice_mem * size;
      solution->parts[i].push_back(Solution::Part{.server = s, .size = size});
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
    solution->servers.push_back(Solution::Server{.mem = server.mem, .net = server.net, .cpu = server.cpu, .socket = server.socket});
  }
  solution->parts.resize(infos.vars.size());
  size_t s = 0;
  size_t total_socket = infos.vars.size();
  double avg_net = infos.avg_net;
  for (size_t i = 0; i < infos.vars.size(); ++i) {
    size_t a = infos.vars[i].slice_num;
    while (a > 0 && s < solution->servers.size()) {
      size_t size;
      if (infos.vars[i].slice_net == 0) {
        size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.vars[i].slice_mem) , a);
        if ((double)(size * solution->servers.size()) / a < 0.2) {
          s++;
          if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
          continue;
        }
      } else {
        size = std::min((size_t)((avg_net - solution->servers[s].net) / infos.vars[i].slice_net) , a);
        size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.vars[i].slice_mem) , size);
        if ((double)(size * solution->servers.size()) / a < 0.2) {
          s++;
          if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
          continue;
        }
      }
      if (infos.vars[i].no_split == true && size != a) {
        s++;
        if (s >= solution->servers.size()) { s = 0; total_socket += solution->servers.size(); avg_net *= 1.1; }
        continue;
      }
      solution->servers[s].mem += infos.vars[i].slice_mem * size;
      solution->servers[s].net += infos.vars[i].slice_net * size;
      solution->servers[s].cpu += infos.vars[i].slice_cpu * size;
      if (infos.vars[i].slice_net != 0) { solution->servers[s].socket++; }
      if (a != size) { total_socket++; }
      size_t iter;
      for (iter = 0; iter < solution->parts[i].size(); ++iter) {
        if (s == solution->parts[i][iter].server) {
          solution->parts[i][iter].size += size;
          break;
        }
      }
      if (iter == solution->parts[i].size()) {
        solution->parts[i].push_back(Solution::Part{.server = s, .size = size});
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

class BalancePlacementer : public Placementer {
 public:
  virtual Status Placement(const std::vector<VariableInfo>& inputs, std::vector<VariableInfo>* outputs, const Arg& arg, size_t server) override {
    VariableInfos infos;
    std::unordered_map<std::string, VariableInfos::Variable> var_map;

    char* meta_var = std::getenv("meta_dir");
    if (meta_var == NULL) { return Status::ArgumentError("MetaInfo DIR Error!"); }
    std::string meta_addr = meta_var;
    std::unique_ptr<FileSystem::ReadStream> s;
    PS_CHECK_STATUS(FileSystem::OpenReadStreamAny(meta_addr, &s));
    LOG(INFO) << "Load Placement Meta Info From: " << meta_addr;
    size_t meta_size;
    size_t max_visit = 0;
    PS_CHECK_STATUS(s->ReadRaw(&meta_size));
    for (size_t i = 0; i < meta_size; ++i) {
      VariableInfos::Variable x;
      std::string meta_data;
      PS_CHECK_STATUS(s->ReadStr(&meta_data));
      std::istringstream is(meta_data);
      is >> x.name >> x.visit_time >> x.dense_visit_ids >> x.sparse_visit_ids >> x.dimension;
      if (x.visit_time > max_visit) { max_visit = x.visit_time; }
      var_map[x.name] = x;
    }

    infos.mem = arg.mem;
    double total_net = 0;
    double total_cpu = 0;
    for (size_t i = 0; i < server; ++i) {
      infos.servers.push_back(VariableInfos::ServerInfo{.mem = 0, .net = 0, .cpu = 0, .socket = 0});
    }
    for (const VariableInfo& info : inputs) {
      auto iter = var_map.find(info.name);
      if (iter == var_map.end()) {
        return Status::ArgumentError("Placement MetaInfo Error!");
      }
      VariableInfos::Variable x = iter->second;
      auto argiter = info.args.find("mem_ratio");
      double mem_ratio = argiter == info.args.end() ? kDefaultMemRatio : atof(argiter->second.c_str());
      double slice_ratio;
      argiter = info.args.find("no_split");
      x.no_split = argiter == info.args.end() ? false : true;
      if (info.type == VariableInfo::kIndex) {
        if (info.shape.empty()) {
          x.slice_num = x.dimension;
          x.slice_mem = SizeOfType(info.datatype) * mem_ratio;
          if (x.visit_time == 0) { slice_ratio = 0; }
          else { slice_ratio = double(x.dense_visit_ids + x.sparse_visit_ids) / max_visit / x.slice_num; }
          x.slice_net = SizeOfType(info.datatype) * slice_ratio;
          x.slice_cpu = 1 * slice_ratio;
        } else {
          size_t slice_size = 1;
          for (size_t i = 1; i < info.shape.size(); i++) {
            slice_size *= info.shape[i];
          }
          x.slice_num = x.dimension;
          x.slice_mem = SizeOfType(info.datatype) * slice_size * mem_ratio;
          if (x.visit_time == 0) { slice_ratio = 0; }
          else { slice_ratio = double(x.dense_visit_ids + x.sparse_visit_ids) / max_visit / x.slice_num; }
          x.slice_net = SizeOfType(info.datatype) * slice_size * slice_ratio;
          x.slice_cpu = x.dense_visit_ids == 0 ? slice_size * slice_ratio : slice_size * kSparseCpuRatio * slice_ratio;
        }
      } else if (info.type == VariableInfo::kHash128 || info.type == VariableInfo::kHash64) {
        if (info.shape.empty()) {
          return Status::ArgumentError("Hash Should at least 1 dim");
        }
        size_t slice_size = 1;
        for (size_t i = 1; i < info.shape.size(); i++) {
          slice_size *= info.shape[i];
        }
        x.slice_num = Hasher::kTargetRange;
        x.slice_mem = double((SizeOfType(info.datatype) * slice_size * mem_ratio + kHashMem) * x.dimension) * 2 / Hasher::kTargetRange;
        if (x.visit_time == 0) { slice_ratio = 0; }
        else { slice_ratio = double(x.dense_visit_ids + x.sparse_visit_ids) / max_visit / x.dimension; }
        x.slice_net = double(SizeOfType(info.datatype) * slice_size * x.dimension * slice_ratio) / Hasher::kTargetRange;
        x.slice_cpu = double(kHashCpuRatio * slice_size * x.dimension * slice_ratio) / Hasher::kTargetRange;
      } else {
        return Status::NotImplemented("Balance Placementer not support type: " + std::to_string(info.type) + " @ " + info.name);
      }
      total_net += x.slice_net * x.slice_num;
      total_cpu += x.slice_cpu * x.slice_num;
      if (info.parts.empty()) {
        infos.vars.push_back(x);
      } else {
        for (auto&& part : info.parts) {
          infos.servers[part.server].net += x.slice_net * part.size;
          infos.servers[part.server].mem += x.slice_mem * part.size;
          infos.servers[part.server].cpu += x.slice_cpu * part.size;
          infos.servers[part.server].socket++;
        }
      }
    }
    infos.avg_net = total_net / server;
    infos.avg_cpu = total_cpu / server;
    Solution solution;
    if (infos.vars.size() > 0) {
      bool result = Get(infos, &solution);
      if (!result) {
        return Status::ArgumentError("Cannot Placement, Too Heavy");
      }
    }
    outputs->clear();
    size_t ptr = 0;
    for (VariableInfo info : inputs) {
      if (info.parts.empty()) {
        for (auto item : solution.parts[ptr]) {
          info.parts.push_back(VariableInfo::Part{.server = item.server, .size = item.size});
        }
        ptr++;
      }
      if (!info.shape.empty()) {
        auto iter = var_map.find(info.name);
        info.shape[0] = iter->second.dimension;
      }
      outputs->push_back(info);
    }
    return Status::Ok();
  }
};

PLUGIN_REGISTER(Placementer, Balance, BalancePlacementer);

}
}

