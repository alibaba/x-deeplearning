#include "service/seastar_status.hh"

namespace ps
{
namespace network
{

WaitLock::WaitLock()
{
    sem_init(&mSemaphore, 0, 1);
}

void WaitLock::Wait()
{
    sem_wait(&mSemaphore);
}

void WaitLock::Release()
{
    sem_post(&mSemaphore);
}

WaitLock& WaitLock::GetInstance()
{
    static std::mutex m;
    static std::unique_ptr<WaitLock> w;
    if (!w)
    {
        std::lock_guard<std::mutex> g(m);
        if (!w) 
        { 
            w.reset(new WaitLock()); 
        }
    }
    return *w;
}

bool IsConnectionBroken(int32_t eno)
{
    return eno == ECONNRESET || eno == EBADF || eno == EPIPE || eno == ENOTSOCK;
}

} // namespace network
} // namespace ps
