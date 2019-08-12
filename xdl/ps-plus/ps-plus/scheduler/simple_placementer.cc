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

#include <random>
#include <map>
#include <vector>
#include <algorithm>

namespace ps {
namespace scheduler {

namespace {

struct Variable {
  size_t slice_count;
  double slice_mem, slice_net;
  std::map<int64_t, int64_t> placement;
};

struct ServerInfo {
  size_t used_net;
  size_t used_mem;
  size_t used_query;
};

int64_t nextRandom(int64_t max) {
  static std::mt19937 re;
  return re() % max;
}

bool tryOnePlacement(std::vector<Variable>& vars, const std::vector<ServerInfo>& servers, int64_t net, int64_t mem, int64_t query) { // limited by 1 query
  static const int p = 80; // percent
  int64_t server = servers.size();
  std::vector<int64_t> used_net;
  std::vector<int64_t> used_mem;
  std::vector<int64_t> used_query;
  for (const ServerInfo& info : servers) {
    used_net.push_back(info.used_net);
    used_mem.push_back(info.used_mem);
    used_query.push_back(info.used_query);
  }

  std::map<int64_t, std::vector<int64_t>> query_list;
  for (int64_t i = 0; i < server; i++) {
    query_list[used_query[i]].push_back(i);
  }

  std::vector<std::pair<double, Variable*>> hint_var_pair;
  for (auto& var : vars) {
    hint_var_pair.emplace_back(std::min(var.slice_mem * var.slice_count / double(mem), var.slice_net * var.slice_count / double(net)), &var);
  }
  std::sort(hint_var_pair.begin(), hint_var_pair.end());

  bool success = true;
  for (auto it : hint_var_pair) {
    auto& var = *it.second;
    int64_t count = var.slice_count;
    var.placement.clear();
    while (count > 0) {
      auto iter = query_list.begin();
      if (iter == query_list.end()) {
        success = false;
        break;
      }

      while (true) {
        auto iter2 = iter;
        iter2++;
        if (iter2 == query_list.end() || nextRandom(100) < p) {
          break;
        } else {
          iter = iter2;
        }
      }
      auto& list = iter->second;
      int64_t id = nextRandom(list.size());
      int64_t x = list[id];
      int64_t y = iter->first;
      list[id] = list.back();
      list.pop_back();
      if (list.empty()) {
        query_list.erase(iter);
      }

      int64_t max = int64_t(std::min(double(mem - used_mem[x]) / var.slice_mem, double(net - used_net[x]) / var.slice_net));
      max = max < count ? max : count;
      var.placement[x] += max;
      count -= max;
      used_mem[x] += var.slice_mem * max;
      used_net[x] += var.slice_net * max;
      used_query[x] = y + 1;
      if (used_mem[x] < mem && used_net[x] < net && used_query[x] < query) {
        query_list[y + 1].push_back(x);
      }
    }
    if (!success) {
      break;
    }
  }
  return success;
}

bool tryPlacement(std::vector<Variable>& vars, const std::vector<ServerInfo>& servers, int64_t net, int64_t mem, int64_t query) { // limited by 1 query
  for (int i = 0; i < 10; i++) {
    if (tryOnePlacement(vars, servers, net, mem, query)) {
      return true;
    }
  }
  return false;
}

int64_t getPlacement(std::vector<Variable>& vars, const std::vector<ServerInfo>& servers, int64_t net, int64_t mem, int64_t query) { // limited by 1 second
  int64_t max_qps = query;
  for (const ServerInfo& server : servers) {
    if (server.used_net > 0) {
      max_qps = std::min(int64_t(net / server.used_net), max_qps);
    }
    if (server.used_mem > 0) {
      max_qps = std::min(int64_t(mem / server.used_mem), max_qps);
    }
    max_qps = std::min(int64_t(query - server.used_query), max_qps);
  }
  auto x = vars;
  int64_t beg = 1, end = max_qps;
  while (beg < end) {
    int64_t mid = (beg + end) / 2 + 1;
    if (tryPlacement(x, servers, net / mid, mem, query / mid)) {
      beg = mid;
      vars = x;
    } else {
      end = mid - 1;
    }
  }
  return beg;
}

}

class SimplePlacementer : public Placementer {
 public:
  virtual Status Placement(const std::vector<VariableInfo>& inputs, std::vector<VariableInfo>* outputs, const Arg& arg, size_t server) override {
    std::vector<Variable> vars;
    std::vector<ServerInfo> servers;
    for (size_t i = 0; i < server; i++) {
      servers.push_back(ServerInfo{.used_net = 0, .used_mem = 0, .used_query = 0});
    }
    for (const VariableInfo& info : inputs) {
      Variable x;
      if (info.type == VariableInfo::kIndex) {
        if (info.shape.empty()) {
          x.slice_count = 1;
          x.slice_mem = SizeOfType(info.datatype);
          x.slice_net = SizeOfType(info.datatype);
        } else {
          size_t slice_size = 1;
          for (size_t i = 1; i < info.shape.size(); i++) {
            slice_size *= info.shape[i];
          }
          auto iter = info.args.find("batch_read");
          double ratio = iter == info.args.end() ? 1 : atof(iter->second.c_str()) / info.shape[0];
          x.slice_count = info.shape[0];
          x.slice_mem = SizeOfType(info.datatype) * slice_size;
          x.slice_net = SizeOfType(info.datatype) * slice_size * ratio;
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
        x.slice_count = Hasher::kTargetRange;
        x.slice_mem = double(SizeOfType(info.datatype) * slice_size * info.shape[0]) / Hasher::kTargetRange;
        x.slice_net = double(SizeOfType(info.datatype) * slice_size * batch_read) / Hasher::kTargetRange;
      } else {
        return Status::NotImplemented("Simple Placementer not support type: " + std::to_string(info.type) + " @ " + info.name);
      }
      if (info.parts.empty()) {
        vars.push_back(x);
      } else {
        for (auto&& part : info.parts) {
          servers[part.server].used_net += x.slice_net * part.size;
          servers[part.server].used_mem += x.slice_mem * part.size;
          servers[part.server].used_query += 1;
        }
      }
    }
    size_t result = getPlacement(vars, servers, arg.net, arg.mem, arg.query);
    if (result == 1) {
      return Status::ArgumentError("Cannot Placement, Too Heavy");
    }
    outputs->clear();
    size_t ptr = 0;
    for (VariableInfo info : inputs) {
      if (info.parts.empty()) {
        for (auto item : vars[ptr].placement) {
          if (item.second > 0) {
            info.parts.push_back(VariableInfo::Part{.server = (size_t)item.first, .size = (size_t)item.second});
          }
        }
        ptr++;
      }
      outputs->push_back(info);
    }
    return Status::Ok();
  }
};

PLUGIN_REGISTER(Placementer, Default, SimplePlacementer);

}
}

