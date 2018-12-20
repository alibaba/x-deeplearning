/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PS_COMMON_ZK_WRAPPER_H
#define PS_COMMON_ZK_WRAPPER_H


#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>

#define THREADED
#include <zookeeper.h>
#undef THREADED

namespace ps {

class ZkWrapper
{
public:
    enum ZkStatus
    {
        ZK_BAD = 0,
        ZK_CONNECTING,
        ZK_CONNECTED
    };
    typedef std::function<void (ZkWrapper *, const std::string &, ZkStatus)> CallbackFuncType;

    ZkWrapper(const std::string & host="", unsigned int timeout = 10000);
    ~ZkWrapper();

private:
    ZkWrapper(const ZkWrapper &);
    ZkWrapper& operator = (const ZkWrapper &);

public:
    bool Open();
    void Close();
    bool IsConnected() const;
    bool IsConnecting() const;
    bool IsBad() const;
    ZkStatus GetStatus() const;
    static ZkStatus State2Status(int state);

    // NOTICE: ZkWrapper object should not be deleted in callback, which will cause dead lock!!!
    void SetConnCallback(const CallbackFuncType &callback);
    void SetChildCallback(const CallbackFuncType &callback);
    void SetDataCallback(const CallbackFuncType &callback);
    void SetCreateCallback(const CallbackFuncType &callback);
    void SetDeleteCallback(const CallbackFuncType &callback);
    
    bool Touch(const std::string &path, const std::string & value, bool permanent = false);
    bool CreatePath(const std::string &path);
    bool GetChild(const std::string &path, std::vector<std::string> &vString, bool watch=false);
    bool GetData(const std::string &path, std::string &str, bool watch=false);
    bool Check(const std::string &path, bool &bExist, bool watch = false);
    bool Remove(const std::string &path);
    bool SetData(const std::string &path, const std::string &value);
    bool TouchSeq(const std::string &basePath, std::string &resultPath, const std::string &value, bool permanent=false);
    int GetState() const;

private:
    static void Watcher(zhandle_t*, int type, int state, const char* path, void* context);

    bool Connect();


    bool CreateNode(const std::string &path, const std::string &value, bool permanent=false);
    bool CreateSeqNode(const std::string &path, std::string &resultPath, 
                       const std::string &value, bool permanent=false);

    bool DeleteNode(const std::string &path);
    bool SetNode(const std::string &path, const std::string &str);
    bool CreateParentPath(const std::string &path);
    bool Validate(const std::string& path);

private:
    zhandle_t *mZkHandle;

    std::string mHost;
    unsigned int mTimeout;

    CallbackFuncType mConnCallback;
    CallbackFuncType mChildCallback;
    CallbackFuncType mDataCallback;
    CallbackFuncType mCreateCallback;
    CallbackFuncType mDeleteCallback;

    std::mutex mMutex;
    std::mutex mClosingMutex;
};

} // namespace ps

#endif //PS_COMMON_ZK_WRAPPER_H
