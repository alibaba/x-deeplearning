#ifndef ML_PS5_MESSAGE_SERIALIZER_H
#define ML_PS5_MESSAGE_SERIALIZER_H

#include <vector>
#include <stdexcept>
#include <utility>
#include "message_header.hh"
#include "fragment.hh"

namespace ps
{
namespace coding
{

class MessageSerializer
{
public:
    MessageSerializer()
    {
    }

    virtual ~MessageSerializer() { }

    virtual void Serialize() = 0;

#ifdef USE_STATISTICS
    MessageHeader::TimeStamp* GetTimeStamp() { return &mMessageHeader.mTimeStamp; }
#endif

    uint64_t GetCalculateCostTime() const { return mMessageHeader.mCalculateCostTime; }
    void SetCalculateCostTime(uint64_t t) { mMessageHeader.mCalculateCostTime = t; }

    uint64_t GetSequence() { return mMessageHeader.mSequence; }
    void SetSequence(uint64_t seq) { mMessageHeader.mSequence = seq; }

    int32_t GetUserThreadId() const { return mMessageHeader.mUserThreadId; }
    void SetUserThreadId(int32_t userId) { mMessageHeader.mUserThreadId = userId; }

    uint64_t GetProcessorClassId() const { return mMessageHeader.mProcessorClassId; }
    void SetProcessorClassId(uint64_t classId) { mMessageHeader.mProcessorClassId = classId; }

    void AppendMeta(const void* data, size_t length)
    {
        const char* const ptr = reinterpret_cast<const char*>(data);
        mMetaBuffer.insert(mMetaBuffer.end(), ptr, ptr + length);
    }

    void AppendMeta(size_t value)
    {
        AppendMeta(&value, sizeof(value));
    }

    void AppendFragment(const void* data, size_t length)
    {
        Fragment frag;
        frag.base = static_cast<char*>(const_cast<void*>(data));
        frag.size = length;
        AppendFragment(frag);
    }

    void AppendFragment(const Fragment& fragment)
    {
        mFragments.push_back(fragment);
    }

    void AppendFragments(const std::vector<Fragment>& fragments)
    {
        mFragments.insert(mFragments.end(), fragments.begin(), fragments.end());
    }

    std::vector<Fragment>&& GetFragments()
    {
        ReserveForHeaders();
        Serialize();
        PatchHeaders();
        PatchFragments();
        return std::move(mFragments);
    }

private:
    void ReserveForHeaders()
    {
        mMetaBuffer.clear();
        mFragments.clear();
        AppendFragment(NULL, 0);  /* hole for mMessageHeader */
        AppendFragment(NULL, 0);  /* hole for mMetaBuffer */
    }

    void PatchHeaders()
    {
        if (GetProcessorClassId() == 0)
        {
            throw std::runtime_error("Processor class id is not specified.");
        }
        mMessageHeader.mMetaBufferSize = mMetaBuffer.size();
        mMessageHeader.mDataBufferSize = 0;
        for (size_t i = 2; i < mFragments.size(); ++i)
        {
            mMessageHeader.mDataBufferSize += mFragments.at(i).size;
        }
    }

    void PatchFragments()
    {
        mFragments[0].base = reinterpret_cast<char*>(&mMessageHeader);
        mFragments[0].size = sizeof(mMessageHeader);
        mFragments[1].base = mMetaBuffer.data();
        mFragments[1].size = mMetaBuffer.size();
    }

    MessageHeader mMessageHeader;
    std::vector<char> mMetaBuffer;
    std::vector<Fragment> mFragments;
};

} // namespace coding
} // namespace ps

#endif // ML_PS5_MESSAGE_SERIALIZER_H
