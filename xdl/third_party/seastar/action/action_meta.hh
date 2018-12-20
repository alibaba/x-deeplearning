#ifndef PS_ACTION_ACTION_META_H
#define PS_ACTION_ACTION_META_H

#include <stdint.h>
#include <iostream>
#include <sstream>

namespace ps
{
namespace action
{
using namespace std;

enum PushPullType
{
    kPushWithRes = 0,
    kPushNoRes,
    kPull
};

struct ActionMeta
{
    ActionMeta() : kType(kPushWithRes), kDataType(0) {}
    ActionMeta(enum PushPullType v, int32_t t) : kType(v), kDataType(t) {}
    ~ActionMeta() {}
    const enum PushPullType kType;
    const int32_t kDataType;
    string ToString() const
    {
        ostringstream sout;
        sout << "kType:" << kType << " kDataType:" << kDataType;
        return sout.str();
    }
};

class ActionMetaHelper
{
public:
    template<typename T> static ActionMeta Get(enum PushPullType v);
};

} // namespace
} // namespace

#endif // PS_ACTION_ACTION_META_H
