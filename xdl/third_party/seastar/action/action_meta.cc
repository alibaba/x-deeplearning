#include <stdint.h>
#include <iostream>
#include "action/action_meta.hh"

namespace ps
{
namespace action
{
using namespace std;

template<> ActionMeta ActionMetaHelper::Get<int32_t>(enum PushPullType v) { return ActionMeta(v, 1); }
template<> ActionMeta ActionMetaHelper::Get<int64_t>(enum PushPullType v) { return ActionMeta(v, 2); }
template<> ActionMeta ActionMetaHelper::Get<float>(enum PushPullType v) { return ActionMeta(v, 3); }
template<> ActionMeta ActionMetaHelper::Get<double>(enum PushPullType v) { return ActionMeta(v, 4); }
template<> ActionMeta ActionMetaHelper::Get<uint32_t>(enum PushPullType v) { return ActionMeta(v, 5); }
template<> ActionMeta ActionMetaHelper::Get<uint64_t>(enum PushPullType v) { return ActionMeta(v, 6); }

} // namespace
} // namespace
