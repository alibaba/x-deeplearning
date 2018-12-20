#ifndef PS_COMMON_H_
#define PS_COMMON_H_

#include <unordered_map>

namespace ps {
namespace service {
namespace seastar {

class Closure {
public:
    virtual ~Closure() {}
    virtual void Run() = 0;
};

class ClosureManager {
public:
    static std::unordered_map<uint64_t, Closure*>& GetClosureMap() {
        static thread_local std::unordered_map<uint64_t, Closure*> cm;
        return cm;
    }
    static Closure* GetClosure(int64_t seq) {
        auto result = GetClosureMap().find(seq);
        if (result == GetClosureMap().end()) {
            return NULL;
        }
        Closure* cs = result->second;
        GetClosureMap().erase(result);
        return cs;
    }
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_COMMON_H_
