#ifndef PS_ACTION_ACTION_UTILITY_H
#define PS_ACTION_ACTION_UTILITY_H

#include <stdint.h>
#include <iostream>

namespace ps
{
namespace action
{
namespace utility
{
using namespace std;

uint64_t CoutCurrentTimeOnNanoSec(const char * prompt);
uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output);

} // namespace
} // namespace
} // namespace

#endif // PS_ACTION_ACTION_UTILITY_H
