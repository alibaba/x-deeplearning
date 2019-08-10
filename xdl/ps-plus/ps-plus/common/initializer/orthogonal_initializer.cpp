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

#include "orthogonal_initializer.h"

#include <string>
#include <cstring>
#include <cmath>
#include <memory>
#include <Eigen/SVD>  
#include <Eigen/Dense>    

#include "ps-plus/common/types.h"
#include "random/random.h"
#include "random/random_ops.h"
#include "normal_initializer.h"

using namespace Eigen;    
using namespace Eigen::internal;    
using namespace Eigen::Architecture;    

namespace ps {
namespace initializer {

OrthogonalInitializer::OrthogonalInitializer(
    int64_t dim, int seed, float gain)
  : dim_(dim)
  , seed_(seed)
  , gain_(gain) {
  normal_initializer_.reset(new NormalInitializer(seed, 0.0, 1.0));
}

bool OrthogonalInitializer::Accept(DataType type) {
  if (type == DataType::kFloat || 
      type == DataType::kDouble) {
    return true;
  }

  return false;
}

void OrthogonalInitializer::Init(void* data, 
                                 DataType type, 
                                 size_t size) {
  if (size % dim_ != 0) {
    printf("error size\n");
    abort();
  }

  normal_initializer_->Init(data, type, size);
  int64_t row = size / dim_;
  int64_t col = dim_;
  if (type == DataType::kFloat) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m((float*)data, row, col);  
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXf u = svd.matrixU() * gain_;
    Eigen::MatrixXf v = svd.matrixV() * gain_;      
    if ((row == u.rows() && col == u.cols()) ||
        (row == u.cols() && col == u.rows())) {
      Eigen::MatrixXf trans_u = u.transpose();
      memcpy(data, trans_u.data(), size * sizeof(float));
    } else if ((row == v.rows() && col == v.cols()) ||
               (row == v.cols() && col == v.rows())) {
      memcpy(data, v.data(), size * sizeof(float));       
    } else {
      printf("svd result shape[%d,%d|%d,%d] not match\n", u.rows(), u.cols(), v.rows(), v.cols());
      abort();
    }
  } else {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m((double*)data, row, col);      
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXd u = svd.matrixU() * gain_;
    Eigen::MatrixXd v = svd.matrixV() * gain_;
    if ((row == u.rows() && col == u.cols()) ||
        (row == u.cols() && col == u.rows())) {
      Eigen::MatrixXd trans_u = u.transpose();
      memcpy(data, trans_u.data(), size * sizeof(double));
    } else if ((row == v.rows() && col == v.cols()) ||
               (row == v.cols() && col == v.rows())) {
      memcpy(data, v.data(), size * sizeof(double));       
    } else {
      printf("svd result shape[%d,%d|%d,%d] not match\n", u.rows(), u.cols(), v.rows(), v.cols());
      abort();
    }
  }
}

Initializer* OrthogonalInitializer::Clone() {
  return new OrthogonalInitializer(
      dim_, seed_, gain_);
}

} //namespace initializer
} //ps

