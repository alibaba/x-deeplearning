/*
 * \file common_helper.cc
 * \brief The fusion common helper utility.
 */
#include "blaze/operator/common_helper.h"

#include "blaze/common/blob.h"
#include "blaze/common/exception.h"
#include "blaze/common/string_util.h"
#include "blaze/graph/graph.h"

namespace blaze {

size_t CommonHelper::GetSliceAxis(const ArgumentHelper* arg) {
  size_t axis;
  if (arg->HasArgument("axes")) {
    std::vector<size_t> axes = arg->GetRepeatedArgument<size_t>("axes");
    BLAZE_CONDITION_THROW(axes.size() == 1, "axes.size()=", axes.size());
    axis = axes[0];
  } else {
    axis = arg->GetSingleArgument<size_t>("axis", 0);
  }
  return axis;
}
  
size_t CommonHelper::GetSliceStart(const ArgumentHelper* arg) {
  size_t start;
  if (arg->HasArgument("starts")) {
    std::vector<size_t> starts = arg->GetRepeatedArgument<size_t>("starts");
    BLAZE_CONDITION_THROW(starts.size() == 1, "starts.size()=", starts.size());
    start = starts[0];
  } else {
    start = arg->GetSingleArgument<size_t>("start", 0);
  }
  return start;
}

size_t CommonHelper::GetSliceEnd(const ArgumentHelper* arg) {
  size_t end;
  if (arg->HasArgument("ends")) {
    std::vector<size_t> ends = arg->GetRepeatedArgument<size_t>("ends");
    BLAZE_CONDITION_THROW(ends.size() == 1, "ends.size()=", ends.size());
    end = ends[0];
  } else {
    end = arg->GetSingleArgument<size_t>("end", 0);
  }
  return end;
}

size_t CommonHelper::GetReduceAxis(const ArgumentHelper* arg) {
  size_t axis;
  if (arg->HasArgument("axes")) {
    std::vector<size_t> axes = arg->GetRepeatedArgument<size_t>("aexs");
    BLAZE_CONDITION_THROW(axes.size() == 1, "axes.size()=", axes.size());
    axis = axes[0];
  } else {
    axis = arg->GetSingleArgument<size_t>("axis", 0);
  }
  return axis;
}

size_t NElemFromDim(const TensorShape& shape) {
  size_t size = 1;
  for (auto dim : shape.dims()) {
    size *= dim;
  }
  return size;
}

int GetIndicatorLevel(const std::string& indicator_name) {
  auto splits = Split(indicator_name, kSparseFeatureSep);
  CHECK_EQ(splits.size(), 2, "splits.size()=", splits.size());
  return std::stoi(splits[1].c_str());
}

InputType GetSparseInputType(const std::string& input_name) {
  auto splits = Split(input_name, blaze::kSparseFeatureSep);
  CHECK_TRUE(splits.size() <= 2, "feed_name=", input_name, " format error");

  if (strcmp(splits[1].c_str(), blaze::kIdSuffix + 1) == 0) {
    return kInputSparseIds;
  } else if (strcmp(splits[1].c_str(), blaze::kValueSuffix + 1) == 0) {
    return kInputSparseValues;
  } else if (strcmp(splits[1].c_str(), blaze::kIdNumSuffix + 1) == 0) {
    return kInputSparseSegments;
  } else {
    BLAZE_THROW("Unkown sparse input type according to input_name: ", input_name);
  }
}

std::string GetSparseFeatureName(const std::string& input_name) {
  auto splits = Split(input_name, blaze::kSparseFeatureSep);
  return splits[0];
}

}  // namespace blaze

