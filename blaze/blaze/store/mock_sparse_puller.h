/*
 * \file mock_sparse_puller.h 
 * \brief The mock sparse puller.
 */
#pragma once

#include <stdio.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "blaze/store/sparse_puller.h"

namespace blaze {
namespace store {

class MockSparsePuller : public SparsePuller {
 public:
  virtual Status Load(const std::string& url) { return kOK; }
  virtual Status Get(const std::vector<SparsePullerInput>& input,
                     std::vector<SparsePullerOutput>& output);

 protected:
  Status Get(const SparsePullerInput& input, SparsePullerOutput& output);
};

}  // namespace store
}  // namespace blaze
