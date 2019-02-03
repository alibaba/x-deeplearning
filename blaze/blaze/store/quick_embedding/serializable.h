/*!
 * \file serialize.h
 * \desc Serializable interface
 */
#pragma once

#include <iostream>

namespace blaze {
namespace store {

class Serializable {
 public:
  virtual ~Serializable() = default;

  // object byte array size
  virtual uint64_t ByteArraySize() const = 0;

  // dump object to serialized file
  virtual bool Dump(std::ostream *os) const = 0;

  // load object from serialized file
  virtual bool Load(std::istream *is) = 0;
};

}  // namespace store
}  // namespace blaze

