#ifndef PS_NETWORK_BASE_CONNECTION_HH_
#define PS_NETWORK_BASE_CONNECTION_HH_

#include "net/api.hh"
#include "core/iostream.hh"
#include "service/session_context.hh"
#include "service/common.hh"

namespace ps
{
namespace network
{

enum RoleType
{
    Server,
    Client
};

class BaseConnection
{
public:
    BaseConnection(RoleType role, seastar::connected_socket&& fd, int64_t id = -1)
        : mIn(fd.input())
        , mOut(fd.output())
        , mFd(std::move(fd))
        , mIsConnRunning(true)
        , mConnId(id)
        , mRole(role)
        , mSessionContext(NULL) {}

    virtual ~BaseConnection() {}

    SessionContext* GetSessionContext()
    {
        return mSessionContext;
    }

    void SetSessionContext(SessionContext* sc)
    {
        mSessionContext = sc;
    }

    void ResetSessionContext()
    {
        mSessionContext = NULL;
    }

    bool GetConnStatus() { return mIsConnRunning; }
    void SetConnStatus(bool status) { mIsConnRunning = status; }

    int64_t GetConnId() { return mConnId; }

    void DeleteSessionContext()
    {
        if (mSessionContext != NULL)
        {
            ServerConnctionManager::EraseConnection(mSessionContext);
            delete mSessionContext;
            mSessionContext = NULL;
        }
    }

    RoleType GetRoleType()
    {
        return mRole;
    }

    // return the truely send count so far of mOut
    uint64_t GetTruelySendCount()
    {
        return mOut.get_truely_send_count();
    }
    void ResetTruelySendCount()
    {
        mOut.reset_truely_send_count();
    }
public:
    seastar::input_stream<char> mIn;
    seastar::output_stream<char> mOut;

private:
    seastar::connected_socket mFd;
    bool mIsConnRunning;
    uint64_t mConnId;
    RoleType mRole;

protected:
    SessionContext* mSessionContext;
};

} // namespace network
} // namespace ps

#endif // PS_NETWORK_BASE_CONNECTION_HH_
