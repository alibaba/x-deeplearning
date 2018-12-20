#include <stdint.h>
#include <iostream>
#include <time.h>
#include "action/utility.hh"

namespace ps
{
namespace action
{
namespace utility
{
using namespace std;

uint64_t CoutCurrentTimeOnNanoSec(const char * prompt)
{
    return CoutCurrentTimeOnNanoSec(prompt, false);
}
uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output)
{
    static thread_local uint64_t lastCoutTime = 0;
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    uint64_t current = (uint64_t)start_time.tv_sec * 1000000000UL + start_time.tv_nsec;
    if (output)
    {
        std::cout << prompt << ":" << current << "||||||diff:" << current - lastCoutTime << std::endl;
    }
    lastCoutTime = current;
    return current;
}

} // namespace
} // namespace
} // namespace
