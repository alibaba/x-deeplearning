#include <stdint.h>
#include <iostream>
#include "action/push_response_item.hh"

namespace ps
{
namespace action
{
using namespace std;

int32_t PushResponseItem::sCurrentPreTimes = 0;
vector<int64_t> PushResponseItem::sStatVec;
vector<int64_t> PushResponseItem::sMaxVec;
vector<int64_t> PushResponseItem::sMinVec;

} // namespace action
} // ps
