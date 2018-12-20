#ifndef ML_PS5_MESSAGE_PROCESSOR_FACTORY_H
#define ML_PS5_MESSAGE_PROCESSOR_FACTORY_H

#include <string>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include "core/ps_coding/message_processor.hh"

namespace ps
{
namespace coding
{

class MessageProcessorFactory
{
public:
    typedef MessageProcessor* MessageProcessorCreator();

    static MessageProcessorFactory& GetInstance()
    {
        static MessageProcessorFactory sInstance;
        return sInstance;
    }

    void RegisterCreator(uint64_t classId, const char* className,
                         MessageProcessorCreator* creator)
    {
        if (mCreatorMap.count(classId))
        {
            std::ostringstream sout;
            sout << "MessageProcessor '" << className << "' with class id ";
            sout << classId << " (0x" << std::hex << classId << ") ";
            sout << "has been registered; please use another id.";
            throw std::runtime_error(sout.str());
        }
        mCreatorMap[classId] = std::make_pair(className, creator);
    }

    MessageProcessor* CreateInstance(uint64_t classId)
    {
        if (!mCreatorMap.count(classId))
        {
            std::ostringstream sout;
            sout << "MessageProcessor with class id ";
            sout << classId << " (0x" << std::hex << classId << ") ";
            sout << "is not registered.";
            throw std::runtime_error(sout.str());
        }
        MessageProcessorCreator* creator = mCreatorMap[classId].second;
        MessageProcessor* processor = creator();
        return processor;
    }

private:
    typedef std::pair<const char*, MessageProcessorCreator*> PairType;
    typedef std::unordered_map<uint64_t, PairType> MapType;

    MapType mCreatorMap;
};

#define PS_REGISTER_MESSAGE_PROCESSOR(className)                     \
    static struct className##Register                                \
    {                                                                \
        className##Register()                                        \
        {                                                            \
            ps::coding::MessageProcessorFactory::GetInstance().RegisterCreator(  \
                k##className##_ClassId, #className,                  \
                &className##Register::CreateInstance);               \
        }                                                            \
                                                                     \
        static ps::coding::MessageProcessor* CreateInstance()                    \
        {                                                            \
            return new className;                                    \
        }                                                            \
    } MessageProcessorFactory_Register##className;                   \
    /**/

} // namespace coding
} // namespace ps

#endif // ML_PS5_MESSAGE_PROCESSOR_FACTORY_H
