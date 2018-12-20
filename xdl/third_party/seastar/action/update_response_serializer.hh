#ifndef PS_ACTION_ACTION_UPDATE_RESPONSE_SERIALIZER_H
#define PS_ACTION_ACTION_UPDATE_RESPONSE_SERIALIZER_H

#include <stdint.h>
#include <iostream>
#include "core/ps_coding/message_serializer.hh"
#include "action/action_meta.hh"
#include "action/response_processor.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class UpdateResponseSerializer : public ps::coding::MessageSerializer
{
public:
    UpdateResponseSerializer()
    {}
    void Serialize() override
    {
        SetProcessorClassId(kActionResponseProcessor_ClassId);
    }
private:
};

} // namespace
} // namespace

#endif // PS_ACTION_ACTION_UPDATE_RESPONSE_SERIALIZER_H
