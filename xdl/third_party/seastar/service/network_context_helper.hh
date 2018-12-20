#ifndef PS_NETWORK_NETWORK_CONTEXT_HELPER_H
#define PS_NETWORK_NETWORK_CONTEXT_HELPER_H

#include <iostream>
#include <stdint.h>
#include "service/network_context.hh"

namespace ps
{
namespace network
{
using namespace std;

extern int32_t gServerCount;
extern int32_t gCoreCount;
extern int32_t gUserCount;

class NetworkContextHelper
{
public:
    static NetworkContextHelper& GetInstance()
    {
        static NetworkContextHelper sInstance;
        return sInstance;
    }

    static ClientNetworkContext& GetNetworkContext()
    {
        // TODO: flag system, as google flag
        static ClientNetworkContext clientNetworkContext(
                //int32_t serverCount, int32_t clientCount, int32_t coreCount, int32_t userThreadCount,
                gServerCount, 0, gCoreCount, gUserCount, "SEASTAR");
        return clientNetworkContext;
    }
private:
    const int32_t kServerCount = 1;
    const int32_t kCoreCount = 1;
    const int32_t kUserCount = 1;
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

} // namespace network
} // namespace ps

#endif // PS_NETWORK_NETWORK_CONTEXT_HELPER_H
