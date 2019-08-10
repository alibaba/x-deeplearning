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
#include <vector>
#include <set>
#include <random>

namespace ps {
namespace scheduler {

static const double kHashMem = 128;
static const double kDefaultMemRatio = 2;

namespace {

struct VariableInfos {
  struct Variable {
    double slice_mem;
    double slice_net;
    size_t slice_size;
  };

  struct ServerInfo {
    double mem;
    double net;
    double qps;
  };

  std::vector<Variable> vars;
  std::vector<ServerInfo> servers;
  size_t mem;
  size_t net;
  size_t qps;
};

int64_t nextRandom(int64_t max) {
  static std::mt19937 re;
  return re() % max;
}

struct Solution {
  struct Part {
    size_t server;
    size_t size;
  };
  struct Server {
    double mem;
    double net;
    double qps;
    std::set<size_t> variables;
  };
  std::vector<std::vector<Part>> parts;
  std::vector<Server> servers;
};

struct ChangeMove {
  size_t var_id;
  size_t part;
};

bool TryChangeMove(const VariableInfos& infos, Solution* solution) {
  std::vector<ChangeMove> moves;
  for (size_t i = 0; i < solution->parts.size(); i++) {
    std::vector<size_t> part;
    if (solution->parts[i].size() < 2) {
      continue;
    }
    for (size_t j = 0; j < solution->parts[i].size(); j++) {
      if (solution->servers[solution->parts[i][j].server].mem < infos.mem) {
        moves.push_back(ChangeMove{.var_id = i, .part = j});
      }
    }
  }
  if (moves.size() == 0) {
    return false;
  }
  size_t x = nextRandom(moves.size());
  size_t var_id = moves[x].var_id;
  size_t part2 = moves[x].part;
  size_t part1 = nextRandom(solution->parts[var_id].size() - 1);
  if (part1 >= part2) {
    part1++;
  }
  size_t server1 = solution->parts[var_id][part1].server;
  size_t server2 = solution->parts[var_id][part2].server;
  size_t maxsize = std::min((size_t)((infos.mem - solution->servers[server2].mem) / infos.vars[var_id].slice_mem + 1), solution->parts[var_id][part1].size);
  size_t size = 0;
  if (nextRandom(100) < 20 || maxsize == 1) {
    // 20% to move all
    size = maxsize;
  } else {
    size = nextRandom(maxsize - 1) + 1;
  }
  solution->servers[server2].mem += infos.vars[var_id].slice_mem * size;
  solution->servers[server2].net += infos.vars[var_id].slice_net * size;
  solution->parts[var_id][part2].size += size;
  solution->servers[server1].mem -= infos.vars[var_id].slice_mem * size;
  solution->servers[server1].net -= infos.vars[var_id].slice_net * size;
  if (size == solution->parts[var_id][part1].size) {
    solution->servers[server1].qps -= 1;
    solution->servers[server1].variables.erase(var_id);
    solution->parts[var_id][part1] = solution->parts[var_id].back();
    solution->parts[var_id].pop_back();
  } else {
    solution->parts[var_id][part1].size -= size;
  }
  return true;
}

bool TryChangeExchange(const VariableInfos& infos, Solution* solution) {
  return false;
}

bool TryChangeCreate(const VariableInfos& infos, Solution* solution) {
  std::vector<size_t> servers;
  for (size_t i = 0; i < infos.servers.size(); i++) {
    if (solution->servers[i].mem < infos.mem && solution->servers[i].qps < infos.vars.size()) {
      servers.push_back(i);
    }
  }
  if (servers.size() == 0) {
    return false;
  }
  size_t server2 = servers[nextRandom(servers.size())];
  std::vector<size_t> vars;
  for (size_t i = 0; i < infos.vars.size(); i++) {
    if (solution->servers[server2].variables.find(i) == solution->servers[server2].variables.end()) {
      vars.push_back(i);
    }
  }
  if (vars.size() == 0) {
    return false;
  }
  size_t var_id = vars[nextRandom(vars.size())];
  size_t part1 = nextRandom(solution->parts[var_id].size());
  size_t server1 = solution->parts[var_id][part1].server;
  size_t maxsize = std::min((size_t)((infos.mem - solution->servers[server2].mem) / infos.vars[var_id].slice_mem + 1), solution->parts[var_id][part1].size);
  size_t size = 0;
  if (nextRandom(100) < 20 || maxsize == 1) {
    // 20% to move all
    size = maxsize;
  } else {
    size = nextRandom(maxsize - 1) + 1;
  }
  solution->servers[server2].mem += infos.vars[var_id].slice_mem * size;
  solution->servers[server2].net += infos.vars[var_id].slice_net * size;
  solution->servers[server2].qps += 1;
  solution->servers[server2].variables.insert(var_id);
  solution->parts[var_id].push_back(Solution::Part{.server = server2, .size = size});
  solution->servers[server1].mem -= infos.vars[var_id].slice_mem * size;
  solution->servers[server1].net -= infos.vars[var_id].slice_net * size;
  if (size == solution->parts[var_id][part1].size) {
    solution->servers[server1].qps -= 1;
    solution->servers[server1].variables.erase(var_id);
    solution->parts[var_id][part1] = solution->parts[var_id].back();
    solution->parts[var_id].pop_back();
  } else {
    solution->parts[var_id][part1].size -= size;
  }
  return true;
}

bool TryChange(const VariableInfos& infos, Solution* solution) {
  const int P_MOVE = 80;
  const int P_EXCHANGE = 90;
  const int P_CREATE = 100;
  int p = nextRandom(100);
  if (p < P_MOVE && TryChangeMove(infos, solution)) {
    return true;
  }
  if (p < P_EXCHANGE && TryChangeExchange(infos, solution)) {
    return true;
  }
  if (p < P_CREATE && TryChangeCreate(infos, solution)) {
    return true;
  }
  return false;
}

bool CreateBase(const VariableInfos& infos, Solution* solution) {
  for (auto server : infos.servers) {
    solution->servers.push_back(Solution::Server{.mem = server.mem, .net = server.net, .qps = server.qps});
  }
  solution->parts.resize(infos.vars.size());
  size_t s = 0;
  for (size_t i = 0; i < infos.vars.size(); i++) {
    size_t a = infos.vars[i].slice_size;
    while (a > 0 && s < solution->servers.size()) {
      size_t size = std::min((size_t)((infos.mem - solution->servers[s].mem) / infos.vars[i].slice_mem + 1), a);
      solution->servers[s].mem += infos.vars[i].slice_mem * size;
      solution->servers[s].net += infos.vars[i].slice_net * size;
      solution->servers[s].qps += 1;
      solution->servers[s].variables.insert(i);
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

std::vector<double> Calc(const VariableInfos& infos, Solution* solution) {
  std::vector<double> ret;
  for (auto server : solution->servers) {
    ret.push_back(std::min(infos.net / (server.net + 1), infos.qps / (server.qps + 1)));
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

bool Get(const VariableInfos& infos, Solution* solution) {
  if (!CreateBase(infos, solution)) {
    return false;
  }
  std::vector<double> r1 = Calc(infos, solution);
  for (int t = 0; t < 10000; t++) {
    Solution backup = *solution;
    TryChange(infos, solution);
    std::vector<double> r2 = Calc(infos, solution);
    bool ok = true;
    for (size_t i = 0; i < r1.size(); i++) {
      if (r1[i] < r2[i]) {
        ok = true;
        break;
      }
      if (r1[i] > r2[i]) {
        ok = false;
        break;
      }
    }
    int percent = exp(-(double)t / 5000) * 10000;
    percent = 0;
    if (ok || nextRandom(10000) < percent) {
      r1 = std::move(r2);
    } else {
      *solution = backup;
    }
  }
  return true;
}


}

class AnnealPlacementer : public Placementer {
 public:
  virtual Status Placement(const std::vector<VariableInfo>& inputs, std::vector<VariableInfo>* outputs, const Arg& arg, size_t server) override {
    VariableInfos infos;
    infos.mem = arg.mem;
    infos.net = arg.net;
    infos.qps = arg.query;
    for (size_t i = 0; i < server; i++) {
      infos.servers.push_back(VariableInfos::ServerInfo{.mem = 0, .net = 0, .qps = 0});
    }
    for (const VariableInfo& info : inputs) {
      VariableInfos::Variable x;
      if (info.type == VariableInfo::kIndex) {
        if (info.shape.empty()) {
          auto iter = info.args.find("mem_ratio");
          double mem_ratio = iter == info.args.end() ? kDefaultMemRatio : atof(iter->second.c_str());
          x.slice_size = 1;
          x.slice_mem = SizeOfType(info.datatype) * mem_ratio;
          x.slice_net = SizeOfType(info.datatype);
        } else {
          size_t slice_size = 1;
          for (size_t i = 1; i < info.shape.size(); i++) {
            slice_size *= info.shape[i];
          }
          auto iter = info.args.find("batch_read");
          double ratio = iter == info.args.end() ? 1 : atof(iter->second.c_str()) / info.shape[0];
          iter = info.args.find("mem_ratio");
          double mem_ratio = iter == info.args.end() ? kDefaultMemRatio : atof(iter->second.c_str());
          x.slice_size = info.shape[0];
          x.slice_mem = SizeOfType(info.datatype) * slice_size * mem_ratio;
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
        iter = info.args.find("mem_ratio");
        double mem_ratio = iter == info.args.end() ? kDefaultMemRatio : atof(iter->second.c_str());
        x.slice_size = Hasher::kTargetRange;
        x.slice_mem = double((SizeOfType(info.datatype) * slice_size * mem_ratio + kHashMem) * info.shape[0]) * 2 / Hasher::kTargetRange;
        x.slice_net = double(SizeOfType(info.datatype) * slice_size * batch_read) / Hasher::kTargetRange;
      } else {
        return Status::NotImplemented("Anneal Placementer not support type: " + std::to_string(info.type) + " @ " + info.name);
      }
      if (info.parts.empty()) {
        infos.vars.push_back(x);
      } else {
        for (auto&& part : info.parts) {
          infos.servers[part.server].net += x.slice_net * part.size;
          infos.servers[part.server].mem += x.slice_mem * part.size;
          infos.servers[part.server].qps += 1;
        }
      }
    }
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
      outputs->push_back(info);
    }
    return Status::Ok();
  }
};

PLUGIN_REGISTER(Placementer, Anneal, AnnealPlacementer);

}
}

