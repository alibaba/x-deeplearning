/*!
 * \file optimizer.cc
 * \brief The optimizer.
 */
#include "blaze/optimizer/optimizer.h"

namespace blaze {

Optimizer* Optimizer::Get() {
  static std::shared_ptr<Optimizer> inst(new Optimizer());
  return inst.get();
}

}  // namespace blaze
