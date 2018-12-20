#ifndef GLOBAL_COMMON_H_
#define GLOBAL_COMMON_H_

#include <unordered_map>
#include <iostream>

namespace ps
{
namespace network
{

struct ServerConnctionInfo
{
    //bool alive;
};

class ServerConnctionManager
{
public:
    static std::unordered_map<uint64_t, ServerConnctionInfo>& GetMap()
    {
        static thread_local std::unordered_map<uint64_t, ServerConnctionInfo> sci;
        return sci;
    }

    static bool IsAlive(void* sessionContext)
    {
        uint64_t id = (uint64_t)sessionContext;
        auto result = GetMap().find(id);
        if (result == GetMap().end())
        {
            std::cout << "Record connection is closed. id=" << id << std::endl;
            return false;
        }
        return true;
    }

    static void RecordConnection(void* sessionContext)
    {
        uint64_t id = (uint64_t)sessionContext;
        bool succ = GetMap().insert({id, ServerConnctionInfo()}).second;
        if (!succ)
        {
            std::cout << "Record connection failed! id=" << id << std::endl;
            auto result = GetMap().find(id);
            if (result != GetMap().end())
            {
                std::cout << "Record existed! id=" << id << std::endl;
            }
        }
    }

    static void EraseConnection(void* sessionContext)
    {
        uint64_t id = (uint64_t)sessionContext;
        auto result = GetMap().find(id);
        if (result == GetMap().end()) return;
        GetMap().erase(result);
    }
};

} // namespace network
} // namespace ps

#endif // GLOBAL_COMMON_H_

