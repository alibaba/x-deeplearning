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

#include "xdl/data_io/parser/parse_txt.h"

#include <assert.h>
#include <functional>

#include "xdl/data_io/pool.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

int ParseTxt::Tokenize(const char *ptrs[], size_t lens[], const char *str, size_t len, char c, size_t max_count) {
  if (len == 0) {
    return 0;
  }
  ptrs[0] = str;
  int i;
  for (i = 0; i < max_count; ++i) {
    const char *tok = (const char *)memchr(ptrs[i], c, len);
    if (tok == nullptr) {
      lens[i] = len;
      break;
    }
    XDL_CHECK(i != max_count);
    ptrs[i+1] = tok + 1;
    lens[i] = tok - ptrs[i];
    len -= lens[i] + 1;
    assert(len > 0);
  }
  return i+1;
}


int ParseTxt::Tokenize(const char *str, size_t len, char c, size_t max_count, Closure closure) {
  if (len == 0) {
    return 0;
  }
  size_t i;
  for (i = 0; i < max_count; ++i) {
    const char *tok = (const char *)memchr(str, c, len);
    if (tok == nullptr) {
      closure(str, len, i);
      break;
    }
    XDL_CHECK(i != max_count);
    size_t n = tok - str;
    closure(str, n, i);
    str = tok+1;
    len -= n+1;
    XDL_CHECK(len > 0) << "str=" << str << ", c=" << c << ", len=" << len;
  }
  return i+1;
}

ssize_t ParseTxt::GetSize(const char *str, size_t len) {
  const char *endl = (const char *)memchr(str, '\n', len);
  if (endl == nullptr) {
    return -1;
  }
  return endl - str + 1;
}

SGroup *ParseTxt::Run(const char *str, size_t len) {
  SGroup *ret = nullptr;
  if (str == nullptr || len == 0) {
    last_group_key_.clear();
    ret = last_sgroup_;
    last_sgroup_ = nullptr;
    return ret;
  }
  const char *seg_ptrs[MAX_NUM_SEG];
  size_t seg_lens[MAX_NUM_SEG];

  seg_ptrs[0] = str;
  size_t n = Tokenize(str, len, kSEG, MAX_NUM_SEG,
                      [&seg_ptrs, &seg_lens](const char *s, size_t n, size_t i) mutable {
                        seg_ptrs[i] = s;
                        seg_lens[i] = n;
                      });
  XDL_CHECK(n > 2) << "seg " << n << " should > 2, len=" << len << " str=" << str;

  assert(seg_ptrs[1] != nullptr && seg_lens[1] > 0);
  SGroup *sgroup = last_sgroup_;
  SampleGroup *sg;
  /// group key
  if (last_group_key_.empty() || seg_lens[1] != last_group_key_.size() ||
      last_sgroup_->size_ >= MAX_NUM_SAMPLE_OF_GROUP ||
      strncmp(seg_ptrs[1], last_group_key_.c_str(), seg_lens[1]) != 0) {
    if (last_sgroup_ != nullptr) {
      XDL_LOG(DEBUG) << "will return, key=" << last_group_key_
          << " size=" << last_sgroup_->size_;
    }
    sgroup = SGroupPool::Get()->Acquire();
    last_group_key_.assign(seg_ptrs[1], seg_lens[1]);
    ret = last_sgroup_;
    last_sgroup_ = sgroup;
    sg =  sgroup->New();
  } else {
    sg = sgroup->Get();
  }

  /// sample id
  assert(seg_ptrs[0] != nullptr && seg_lens[0] > 0);
  sg->add_sample_ids(seg_ptrs[0], seg_lens[0]);

  FeatureTable *ft;
  if (sg->feature_tables_size() > 0) {
    ft = sg->mutable_feature_tables(0);
  } else {
    ft = sg->add_feature_tables();
  }

  FeatureLine *fl = ft->add_feature_lines();

  /// kvs
  OnFeatureLine(fl, seg_ptrs[2], seg_lens[2], kSparse);

  /// dense
  OnFeatureLine(fl, seg_ptrs[3], seg_lens[3], kDense);

  Label *l = sg->add_labels();
  /// label
  if (!OnLabel(l, seg_ptrs[4], seg_lens[4])) {
    return nullptr;
  }

  sgroup->Reset();

  /// timestamp
  return ret;
}

inline bool ParseTxt::OnSparse(FeatureValue *fv, const char *s, size_t n) {
  char *end;
  int64_t k = strtol(s, &end, 10);
  if (end - s == n) {
    fv->set_key(k);
    fv->set_value(1);
    return true;
  } else if (end - s > n || end[0] != kKEY) {
    return false;
  }
  fv->set_key(k);
  float v = strtof(end+1, &end);
  if (end - s != n) {
    return false;
  }
  fv->set_value(v);
  return true;
}

inline bool ParseTxt::OnDense(FeatureValue *fv, const char *s, size_t n) {
  char *end;
  float v = strtof(s, &end);
  if (end - s != n) {
    return false;
  }
  fv->add_vector(v);
  return true;
}

/// sf1@k1:v1,k2:v2;sf2@k3,k4
/// df1@v1,v2;df2@v3
bool ParseTxt::OnFeatureLine(FeatureLine *fl, const char *str, size_t len, FeatureType type) {
  size_t n = Tokenize(str, len, kFEA, MAX_NUM_FEA,
                      [this, &fl, type](const char *s, size_t n, size_t i) mutable {
                        Feature *f = fl->add_features();
                        this->OnFeature(f, s, n, type);
                      });
  return true;
}


/// l1,l2
bool ParseTxt::OnLabel(Label *l, const char *str, size_t len) {
  size_t n = Tokenize(str, len, kVAL, MAX_NUM_LAB,
                      [this, &l](const char *s, size_t n, size_t i) mutable {
                        char *end;
                        float v = strtof(s, &end);
                        XDL_CHECK(end == s + n);
                        l->add_values(v);
                      });
  return true;
}

bool ParseTxt::OnFeature(Feature *f, const char *str, size_t len, FeatureType type) {
  /// name
  const char *tok = (const char *)memchr(str, kNAM, len);
  size_t n = tok - str;
  XDL_CHECK(tok != nullptr);
  f->set_type(type);
  f->set_name(str, n);

  str = tok + 1;
  len -= n + 1;

  /// value
  if (type == kSparse) {
    n = Tokenize(str, len, kVAL, MAX_NUM_VAL,
                 [this, &f, type](const char *s, size_t n, size_t i) mutable {
                   FeatureValue *v = f->add_values();
                   this->OnSparse(v, s, n);
                 });
  } else {
    FeatureValue *v = f->add_values();
    n = Tokenize(str, len, kVAL, MAX_NUM_VAL,
                 [this, &v, type](const char *s, size_t n, size_t i) mutable {
                   this->OnDense(v, s, n);
                 });
  }
  return true;
}

}  // namespace xdl
}  // namespace io
