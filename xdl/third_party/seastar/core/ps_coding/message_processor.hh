#ifndef ML_PS5_MESSAGE_PROCESSOR_H
#define ML_PS5_MESSAGE_PROCESSOR_H

#include <vector>
#include <stdint.h>
#include "core/future.hh"
#include "message_header.hh"
#include "temporary_buffer.hh"
#include "service/session_context.hh"

namespace ps
{
namespace coding
{

class MessageProcessor
{
public:
    MessageProcessor() { }

    virtual ~MessageProcessor() { }

    virtual seastar::future<> Process(ps::network::SessionContext*) = 0;

    MessageHeader& GetMessageHeader() { return mMessageHeader; }
    const MessageHeader& GetMessageHeader() const { return mMessageHeader; }

    TemporaryBuffer& GetMetaBuffer() { return mMetaBuffer; }
    const TemporaryBuffer& GetMetaBuffer() const { return mMetaBuffer; }

    TemporaryBuffer& GetDataBuffer() { return mDataBuffer; }
    const TemporaryBuffer& GetDataBuffer() const { return mDataBuffer; }

private:
    MessageHeader mMessageHeader;
    TemporaryBuffer mMetaBuffer;
    TemporaryBuffer mDataBuffer;
};

#define PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(className, classId)  \
    const uint64_t k##className##_ClassId = classId;               \
    /**/

} // namespace coding
} // namespace ps

#endif // ML_PS5_MESSAGE_PROCESSOR_H
