#ifndef ML_PS5_SEASTAR_EXCEPTION_H
#define ML_PS5_SEASTAR_EXCEPTION_H

#include <exception>

namespace ps
{
namespace network
{

// exception when connect failed, there has no status
class ReconnectException : public std::exception 
{
};

class ConnectException : public std::exception
{
};

} // network
} // ps

#endif // ML_PS5_SEASTAR_EXCEPTION_H

