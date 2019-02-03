/*
 * \file c_api_blaze_optimizer.cc
 */
#include "blaze/api/c_api/c_api_blaze_optimizer.h"

#include "blaze/optimizer/optimizer.h"
#include "blaze/api/c_api/c_api_error.h"
#include "blaze/common/proto_helper.h"
#include "blaze/common/exception.h"

using blaze::NetDefHelper;
using blaze::Optimizer;
using blaze::NetDef;

// Convert mxnet model to blaze model
int Blaze_OptimizeBlaze(const char* raw_model_file,  // which is binary.
                        const char* dst_model_file,
                        int binary) {
  auto optimizer = Optimizer::Get();

  NetDefHelper net_def_helper;
  NetDef net_def;

  if (!NetDefHelper::LoadNetDefFromBinaryFile(raw_model_file, &net_def)) {
    Blaze_SetLastErrorString("%s load from binary failed", raw_model_file);
    return -1;
  }
  net_def = optimizer->RunPass(net_def);
  if (binary) {
    if (!NetDefHelper::SaveNetDefToBinaryFile(dst_model_file, &net_def)) {
      Blaze_SetLastErrorString("save binary format %s failed", dst_model_file);
      return -1;
    }
  } else {
    if (!NetDefHelper::SaveNetDefToTextFile(dst_model_file, &net_def)) {
      Blaze_SetLastErrorString("save text format %s failed", dst_model_file);
      return -1;
    }
  }
  return 0;
}

