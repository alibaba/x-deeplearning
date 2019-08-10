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

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/hasher.h"
#include "ps-plus/common/file_system.h"

namespace ps {
namespace server {
namespace udf {

namespace {
struct ListHandle {
  HashMapImpl<int64_t>::NonCocurrentHashTable* list;
  int beg, end;
  int threshold;
  bool is_black;
};
}

class HashBlackWhiteList : public SimpleUdf<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx, const std::vector<std::string>& token_names,
      const std::vector<std::string>& var_names,
      const std::vector<std::string>& dirs, const std::vector<int>& threshold,
      const std::vector<int>& is_black, const std::vector<int>& beg, const std::vector<int>& end) const {
    ctx->GetServerLocker()->ChangeType(QRWLocker::kWrite);
    std::unordered_map<std::string, HashMapImpl<int64_t>*> hashmaps;
    std::unordered_map<std::string, ListHandle> lists;
    StorageManager* manager = ctx->GetStorageManager();
    std::unordered_map<std::string, HashMapImpl<int64_t>*> white_hashmap, black_hashmap;
    for (size_t i = 0; i < token_names.size(); i++) {
      std::string token = token_names[i];
      std::string var = var_names[i];
      Variable* variable;
      PS_CHECK_STATUS(manager->Get(var, &variable));
      auto slicer = dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(variable->GetSlicer());
      if (slicer == nullptr) {
        return Status::ArgumentError("HashBlackWhiteList: Variable Should be a Hash Variable for " + var);
      }
      auto hashmap = dynamic_cast<HashMapImpl<int64_t>*>(slicer->Internal().get());
      if (hashmap == nullptr) {
        return Status::ArgumentError("HashBlackWhiteList: Variable Should be a Hash 64 Variable for " + var);
      }
      hashmaps[token] = hashmap;
      if (is_black[i]) {
        if (black_hashmap[var] == nullptr) {
          black_hashmap[var] = hashmap;
          lists[token].list = hashmap->NewBlackList();
        } else {
          lists[token].list = hashmap->GetBlackList();
        }
      } else {
        if (white_hashmap[var] == nullptr) {
          white_hashmap[var] = hashmap;
          lists[token].list = hashmap->NewWhiteList();
        } else {
          lists[token].list = hashmap->GetWhiteList();
        }
      }
      lists[token].beg = beg[i];
      lists[token].end = end[i];
      lists[token].threshold = threshold[i];
      lists[token].is_black = is_black[i];
    }
    std::vector<std::string> files;
    for (auto&& dir : dirs) {
      std::vector<std::string> filenames;
      PS_CHECK_STATUS(FileSystem::ListDirectoryAny(dir, &filenames));
      for (auto&& file : filenames) {
        files.emplace_back(file);
      }
    }
    for (auto& file : files) {
      LOG(INFO) << "Processing " << file;
      std::unique_ptr<FileSystem::ReadStream> f;
      PS_CHECK_STATUS(FileSystem::OpenReadStreamAny(file, &f));
      while (true) {
        bool eof;
        PS_CHECK_STATUS(f->Eof(&eof));
        if (eof) {
          break;
        }
        std::string token;
        int64_t id;
        double d;
        PS_CHECK_STATUS(f->ReadShortStr(&token));
        PS_CHECK_STATUS(f->ReadRaw(&id));
        PS_CHECK_STATUS(f->ReadRaw(&d));
        auto iter = lists.find(token);
        if (iter == lists.end()) {
          continue;
        }
        int hash = Hasher::Hash64(id);
        auto&& list = iter->second;
        if (hash < list.beg || hash >= list.end) {
          continue;
        }
        if (list.is_black) {
          if (d < list.threshold) {
            list.list->insert(id);
          }
        } else {
          if (d >= list.threshold) {
            list.list->insert(id);
          }
        }
      }
      f->Close();
    }
    for (auto&& item : white_hashmap) {
      size_t s = item.second->FilterByWhiteList();
      LOG(INFO) << "filter " << item.first << " as " << s << " listsize " << item.second->GetWhiteList()->size();
    }
    for (auto&& item : black_hashmap) {
      size_t s = item.second->FilterByBlackList();
      LOG(INFO) << "filter " << item.first << " as " << s << " listsize " << item.second->GetBlackList()->size();
    }
    ctx->GetServerLocker()->ChangeType(QRWLocker::kSimpleRead);
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(HashBlackWhiteList, HashBlackWhiteList);

}
}
}

