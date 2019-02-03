/*
 * \file func.h
 * \brief The basic definition of func 
 */

#ifndef BLAZE_COMMON_FUNC_H_
#define BLAZE_COMMON_FUNC_H_

namespace blaze {

class Net;

using PredictorCallback = std::function<void()>;  

struct AsyncTask {
  AsyncTask(Net* n, Net* pn, const PredictorCallback&& c)
    : net(n), parent_net(pn), cb(std::move(c)) {}

  size_t size() {
    return 1;
  }

  Net* net;
  Net* parent_net;
  PredictorCallback cb; 
};

} // namespace blaze

#endif  // BLAZE_COMMON_FUNC_H_

