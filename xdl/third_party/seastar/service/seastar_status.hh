#ifndef PS5_ML_SEASTAR_STATUS_H
#define PS5_ML_SEASTAR_STATUS_H

#include <stdint.h>
#include <errno.h>
#include <semaphore.h>
#include <mutex>
#include <memory>

namespace ps
{
namespace network
{

class WaitLock
{
public:
    ~WaitLock() {}
    WaitLock();
    void Wait();
    void Release();
    static WaitLock& GetInstance();

private:
    sem_t mSemaphore;
};

enum ConnectionStatus
{
    Normal,
    BadConnection,
    CanNotConnectTo,
    TimeOut,
    Unknow
};

bool IsConnectionBroken(int32_t eno);

} // namespace network
} // namespace ps

#endif // PS5_ML_SEASTAR_STATUS_H
