#include <string.h>
#include <iostream>
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "action/action_meta.hh"
#include "action/request_processor.hh"
#include "action/update_response_serializer.hh"
#include "action/update_process_item.hh"
#include "action/get_process_item.hh"
#include "action/utility.hh"
#include "action/action_meta.hh"
#include "core/reactor.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
ps::network::PsWorkItem* GenerateProcessItem(ps::network::SessionContext* sc,
        ps::coding::MessageHeader* header, ps::coding::TemporaryBuffer* data)
{
    UpdateResponseSerializer<T>* serializer = new UpdateResponseSerializer<T>();
    *serializer->GetTimeStamp() = header->mTimeStamp;
    serializer->GetTimeStamp()->mServerProcessTime
        = utility::CoutCurrentTimeOnNanoSec("___server_process_time", false);
    serializer->SetSequence(header->mSequence);
    serializer->SetUserThreadId(header->mUserThreadId);
    return new ps::action::UpdateProcessItem<T>(0, sc, serializer);
}

seastar::future<> ActionRequestProcessor::Process(ps::network::SessionContext* sc)
{
    ActionMeta meta;
    memcpy(reinterpret_cast<void*>(&meta),
            reinterpret_cast<const void*>(this->GetMetaBuffer().get()),
            sizeof(ActionMeta));
    cout << "ccccccccccccccccccc_meta:" << meta.ToString() << endl;
    ps::network::PsWorkItem* item = NULL;
    if (meta.kType == kPull)
    {
        switch(meta.kDataType)
        {
        case 1:
        {
            item = GenerateProcessItem<int32_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 2:
        {
            item = GenerateProcessItem<int64_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 3:
        {
            item = GenerateProcessItem<float>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 4:
        {
            item = GenerateProcessItem<double>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 5:
        {
            item = GenerateProcessItem<uint32_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 6:
        {
            item = GenerateProcessItem<uint64_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        default:
        {
            break;
        }
        }
    }
    else
    {
        // Push
        switch(meta.kDataType)
        {
        case 1:
        {
            item = GenerateProcessItem<int32_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 2:
        {
            item = GenerateProcessItem<int64_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 3:
        {
            item = GenerateProcessItem<float>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 4:
        {
            item = GenerateProcessItem<double>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 5:
        {
            item = GenerateProcessItem<uint32_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        case 6:
        {
            item = GenerateProcessItem<uint64_t>(meta.kType == kPushWithRes ? sc : NULL,
                    &this->GetMessageHeader(), &this->GetDataBuffer());
            break;
        }
        default:
        {
            break;
        }
        }
    }
    std::pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHubWithoutLock<ps::network::Item>("SEASTAR");
    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    // TODO: multiple shard
    outputHub->Enqueue(item, seastar::engine().cpu_id(), 0);
    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(ActionRequestProcessor);

} // namespace action
} // namespace ps
