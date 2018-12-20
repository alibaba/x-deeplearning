#include "packet_timer.hh"
#include "service/client.hh"
#include "service/session_context.hh"
#include "core/ps_coding/message_processor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "service/client_network_context.hh"
#include <sys/time.h>
#include <iostream>

namespace ps
{
namespace network
{

void PacketTimer::Start()
{
    mTimer.set_callback([this]
    {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        uint64_t now = tv.tv_sec * 1000000 + tv.tv_usec;
        auto requestInfoSet =  mConn->GetRequestInfoSet();
        uint64_t processorId = mConn->GetClientNetworkContext()->GetResponseProcessorIdOfServer(mConn->GetConnId());

        // do something via MxN queue
        if (processorId == 0)
        {
            for(auto iter = requestInfoSet->begin(); iter != requestInfoSet->end();)
            {
                // timeout 1s
                if (now - iter->sendTime < mInterval * 1000)
                {
                    ++iter;
                    continue;
                }

                std::cout << "PacketTimer::Start: seqid=" << iter->sequenceId << ", serverid=" << mConn->GetConnId() << std::endl;
                PsWorkItem* pItem = new PsWorkItem();
                pItem->SetSideSequ(iter->sequenceId);
                pItem->SetStatus(ps::network::ConnectionStatus::TimeOut);
                pItem->SetServerId(mConn->GetConnId());
                unsigned queueId = engine().cpu_id();
                mConn->GetClientNetworkContext()->PushItemBack(pItem, queueId, iter->userThreadId);
                iter = requestInfoSet->erase(iter);
            }
        }
        // do something in processor
        else
        {
            SessionContext* sc = mConn->GetClientNetworkContext()->GetSessionOfId(mConn->GetConnId());
            sc->ClearSavedSequence();
            for(auto iter = requestInfoSet->begin(); iter != requestInfoSet->end();)
            {
                if (now - iter->sendTime < mInterval * 1000)
                {
                    ++iter;
                    continue;
                }

                sc->PutOneToSavedSequence(iter->sequenceId);
                iter = requestInfoSet->erase(iter);
            }
            
            // there are some timeout packets
            if (sc->GetSavedSequenceWhenBroken().size() > 0)
            {
                std::cout << "There are some packets are timeout. Timeout time: " << mInterval << "ms"  << std::endl;
                sc->SetHaveTimeOutPacket(true);
                ps::coding::MessageProcessor* processor =
                    ps::coding::MessageProcessorFactory::GetInstance().
                    CreateInstance(processorId);
                processor->Process(sc).then([processor, sc] ()
                {
                    // consume done
                    sc->SetHaveTimeOutPacket(false);
                    delete processor;
                });
            }
        }
    });

    // per second
    mTimer.arm_periodic(std::chrono::milliseconds(mInterval));
}


} // namespace network
} // namespace ps

