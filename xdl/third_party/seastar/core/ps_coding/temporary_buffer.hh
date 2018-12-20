#ifndef ML_PS5_TEMPORARY_BUFFER_H
#define ML_PS5_TEMPORARY_BUFFER_H

#include "core/temporary_buffer.hh"

namespace ps
{
namespace coding
{

typedef seastar::temporary_buffer<char> TemporaryBuffer;

} // namespace coding
} // namespace ps

#endif // ML_PS5_TEMPORARY_BUFFER_H
