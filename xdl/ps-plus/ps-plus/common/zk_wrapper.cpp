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

#include "ps-plus/common/zk_wrapper.h"

#include "ps-plus/common/string_utils.h"

#include <iostream>
#include "ps-plus/common/logging.h"

#define SEQIDLENGTH 128

using namespace std;

namespace ps {

ZkWrapper::ZkWrapper(const std::string & host, unsigned int timeout)
  : mZkHandle(NULL), mHost(host), mTimeout(timeout)
{
}

ZkWrapper::~ZkWrapper() 
{
  Close();
}

void ZkWrapper::SetConnCallback(const CallbackFuncType &callback) 
{
  std::unique_lock<mutex> lock(mMutex);
  mConnCallback = callback;
}

void ZkWrapper::SetChildCallback(const CallbackFuncType &callback) 
{
  std::unique_lock<mutex> lock(mMutex);
  mChildCallback = callback;
}

void ZkWrapper::SetDataCallback(const CallbackFuncType &callback) 
{
  std::unique_lock<mutex> lock(mMutex);
  mDataCallback = callback;
}

void ZkWrapper::SetCreateCallback(const CallbackFuncType &callback) 
{
  std::unique_lock<mutex> lock(mMutex);
  mCreateCallback = callback;
}

void ZkWrapper::SetDeleteCallback(const CallbackFuncType &callback) 
{
  std::unique_lock<mutex> lock(mMutex);
  mDeleteCallback = callback;
}

bool ZkWrapper::Open() 
{
  Close();
  return Connect();
}

bool ZkWrapper::Connect()
{
  mZkHandle = zookeeper_init(mHost.c_str(), &ZkWrapper::Watcher, mTimeout, 0, this, 0);
  if (NULL == mZkHandle) {
    return false;
  }
  return true;
}

void ZkWrapper::Close()
{
  if (NULL != mZkHandle)
  {
    std::unique_lock<mutex> lock(mClosingMutex);
    zoo_set_context(mZkHandle, NULL);
    zoo_set_watcher(mZkHandle, NULL);
    int ret = zookeeper_close(mZkHandle);
    if(ZOK != ret) {
      LOG(ERROR) << "zookeeper handle close failed.";
    }
    mZkHandle = NULL;
  }
}

bool ZkWrapper::Touch(const std::string &path, const std::string &value, bool permanent)
{
  if (!Validate(path))
  {
    return false;
  }
    
  if (SetNode(path, value))
  {
    return true;
  }
    
  DeleteNode(path);
  if (CreateNode(path, value, permanent))
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool ZkWrapper::CreatePath(const std::string &path)
{
  if (!Validate(path))
  {
    return false;
  }
    
  return CreateParentPath(path + "/1");
}

bool ZkWrapper::SetData(const std::string &path, const std::string &value) 
{
  if (!Validate(path))
  {
    return false;
  }
    
  if (SetNode(path, value))
  {
    return true;
  }
        
  return false;
}

bool ZkWrapper::TouchSeq(const std::string &basePath, std::string &resultPath, 
                         const std::string &value, bool permanent)
{
  if (!Validate(basePath))
  {
    return false;
  }

  if (CreateSeqNode(basePath, resultPath, value, permanent))
  {
    return true;
  }
  else 
  {
    return false;
  }
}

bool ZkWrapper::GetChild(const std::string &path, std::vector<std::string> &vString, bool watch)
{
  vString.clear();
  if (!Validate(path))
  {
    return false;
  }

  struct String_vector strings;
  int ret = zoo_get_children(mZkHandle, path.c_str(), watch ? 1:0, &strings);
  if (ZOK == ret)
  {
    for (int i = 0 ; i < strings.count ; ++i) 
    {
      vString.push_back(std::string(strings.data[i]));
    }
    deallocate_String_vector(&strings);
    return true;
  }
  else 
  {
    LOG(ERROR) << "zookeeper getchild failed. " << zerror(ret) << "(" << ret << ")";
    return false;
  }

  return false;
}

bool ZkWrapper::GetData(const std::string &path, std::string &str, bool watch)
{
  if (!Validate(path))
  {
    return false;
  }

  char buffer[1024];
  struct Stat stat;

  int buffer_len = sizeof(buffer);
  int ret = zoo_get(mZkHandle, path.c_str(), watch?1:0, buffer, &buffer_len, &stat);
  if (ZOK != ret)
  {
    if (ZNONODE != ret)
    {
      LOG(ERROR) << "zookeeper get failed. " << zerror(ret);
    }
    return false;
  }
        
  if ((unsigned)stat.dataLength > sizeof(buffer))
  {
    char *newBuffer = new char[stat.dataLength];
    buffer_len = stat.dataLength;
    int ret = zoo_get(mZkHandle, path.c_str(), watch?1:0, newBuffer, &buffer_len, &stat);
    if(ZOK != ret)
    {
      delete [] newBuffer;
      if(ZNONODE != ret)
      {
        LOG(ERROR) << "zookeeper get failed. " << zerror(ret);
      }
      return false;
    }
    str = std::string(newBuffer, (size_t)buffer_len);
    delete [] newBuffer;
  }
  else 
  {
    str = std::string(buffer, (size_t)buffer_len);
  }
  return true;
}

bool ZkWrapper::Check(const std::string &path, bool &bExist, bool watch)
{
  bExist = false;
  if (!Validate(path))
  {
    return false;
  }

  struct Stat stat;
  int ret = zoo_exists(mZkHandle, path.c_str(), watch?1:0, &stat);
  if (ZOK == ret)
  {
    bExist = true;
    return true;
  }
  else if(ZNONODE == ret)
  {
    bExist = false;
    return true;
  }

  LOG(ERROR) << "zookeeper check exist failed. " << zerror(ret);
  return false;
}

bool ZkWrapper::Remove(const std::string &path) 
{
  if (!Validate(path))
  {
    return false;
  }

  int ret = zoo_delete(mZkHandle, path.c_str(), -1);
  if ((ZOK == ret) || (ZNONODE == ret)) 
  {
    return true;
  }
  else if (ZNOTEMPTY == ret)
  {
    struct String_vector strings;
    ret = zoo_get_children(mZkHandle, path.c_str(), 0, &strings);
    if(ZOK == ret) 
    {
      for(int i = 0 ; i < strings.count ; ++i) 
      {
        Remove(path + "/" + strings.data[i]);
      }            
      deallocate_String_vector(&strings);
    }
    ret = zoo_delete(mZkHandle, path.c_str(), -1);
    if((ZOK == ret) || (ZNONODE == ret)) 
    {
      return true;
    }
  }

  LOG(ERROR) << "zookeeper delete failed. " << zerror(ret);
  return false;
}

void ZkWrapper::Watcher(zhandle_t *zk, int type, int state, 
                        const char *path, void* context)
{
  if(ZOO_NOTWATCHING_EVENT == type) 
  {
  }
  else if (context)
  {
    ZkWrapper *pthis = (ZkWrapper*)context;
    if (!pthis->mClosingMutex.try_lock())
    {
      LOG(INFO) << "watcher event ignored by closing! path: " << path;
      return;
    }
    CallbackFuncType callback;
    bool needCall = true;
    {
      std::unique_lock<mutex> lock(pthis->mMutex);
      if (ZOO_CHANGED_EVENT == type)
      {
        //zoo_get(), child delete or create
        callback = pthis->mDataCallback;
      }
      else if (ZOO_CHILD_EVENT == type)
      {
        //zoo_get_children(), child create or delete
        callback = pthis->mChildCallback;
      }
      else if (ZOO_SESSION_EVENT == type)
      {
        //connect(reconnect) successfully
        callback = pthis->mConnCallback;
      }
      else if (ZOO_CREATED_EVENT == type)
      {
        //zoo_exist()
        callback = pthis->mCreateCallback;
      }
      else if(ZOO_DELETED_EVENT == type)
      {
        //zoo_exist() or zoo_get()
        callback = pthis->mDeleteCallback;
      }
      else 
      {
        needCall = false;
      }
    }
    if (needCall && callback)
    {
      callback(pthis, std::string(path), State2Status(state));
    }
    pthis->mClosingMutex.unlock();
  }
}

int ZkWrapper::GetState() const 
{
  if (!mZkHandle) 
  {
    return ZOO_EXPIRED_SESSION_STATE;
  }
  return zoo_state(mZkHandle);
}

bool ZkWrapper::IsConnected() const 
{
  return (ZK_CONNECTED == GetStatus());
}

bool ZkWrapper::IsConnecting() const 
{
  return (ZK_CONNECTING == GetStatus());
}

bool ZkWrapper::IsBad() const 
{
  return (ZK_BAD == GetStatus());
}

ZkWrapper::ZkStatus ZkWrapper::GetStatus() const 
{
  return State2Status(GetState());
}

ZkWrapper::ZkStatus ZkWrapper::State2Status(int state) 
{
  if (ZOO_CONNECTED_STATE == state)
  {
    return ZK_CONNECTED;
  }
  else if ((ZOO_ASSOCIATING_STATE == state) || (ZOO_CONNECTING_STATE == state)) 
  {
    return ZK_CONNECTING;
  }
  return ZK_BAD;        
}

bool ZkWrapper::CreateNode(const std::string &path, const std::string &value, bool permanent) 
{
  int ret = zoo_create(mZkHandle, path.c_str(), value.c_str(), value.length(), 
                       &ZOO_OPEN_ACL_UNSAFE, permanent?0:ZOO_EPHEMERAL, NULL, 0);
  if (ZOK == ret) 
  {
    return true;
  }
  else if (ZNONODE == ret)
  {
    if (!CreateParentPath(path))
    {
      return false;
    }

    int ret = zoo_create(mZkHandle, path.c_str(), value.c_str(), value.length(), 
                         &ZOO_OPEN_ACL_UNSAFE, permanent?0:ZOO_EPHEMERAL, NULL, 0);
    if(ZOK == ret) 
    {
      return true;
    }
    else 
    {
      LOG(ERROR) << "zookeeper create Node " << path << " failed: " << zerror(ret);
      return false;
    }
  }
  else 
  {
    LOG(ERROR) << "zookeeper create Node " << path << " failed: " << zerror(ret);
    return false;
  }
}

bool ZkWrapper::CreateSeqNode(const std::string &path, std::string &resultPath, 
                              const std::string &value, bool permanent) 
{
  std::vector<char> resultBuff(path.length() + SEQIDLENGTH);
  int ret = zoo_create(mZkHandle, path.c_str(), value.c_str(), value.length(), 
                       &ZOO_OPEN_ACL_UNSAFE, permanent?ZOO_SEQUENCE:(ZOO_SEQUENCE|ZOO_EPHEMERAL), 
                       &*resultBuff.begin(), resultBuff.size());
  if (ZOK == ret)
  {
    resultPath = &*resultBuff.begin();
    return true;
  }
  else if (ZNONODE == ret) 
  {
    if (!CreateParentPath(path))
    {
      return false;
    }

    int ret = zoo_create(mZkHandle, path.c_str(), value.c_str(), value.length(), 
                         &ZOO_OPEN_ACL_UNSAFE, permanent?ZOO_SEQUENCE:(ZOO_SEQUENCE|ZOO_EPHEMERAL), 
                         &*resultBuff.begin(), resultBuff.size());
    if(ZOK == ret)
    {
      resultPath = &*resultBuff.begin();
      return true;
    }
    else 
    {
      LOG(ERROR) << "zookeeper create node " << path << " failed: " << zerror(ret);
      return false;
    }
  }
  else
  {
    LOG(ERROR) << "zookeeper create node " << path << " failed: " << zerror(ret);
    return false;
  }
}

bool ZkWrapper::CreateParentPath(const std::string &path)
{
  std::vector<std::string> paths = StringUtils::split(path,"/");
  paths.pop_back();
  std::string current;
  for (std::vector<std::string>::iterator i = paths.begin() ; i != paths.end() ; ++i) 
  {
    current += std::string("/") + *i;
    bool succ = false;
    int mTryCount = StringUtils::split(mHost, ",").size();
    for (int tryCount = 0; tryCount < mTryCount; ++tryCount) 
    {
      int ret = zoo_create(mZkHandle, current.c_str(), "" , 0,
                           &ZOO_OPEN_ACL_UNSAFE, 0, NULL, 0);
      if (ret == ZOK || ret == ZNODEEXISTS) 
      {
        succ = true;
        break;
      }

      if (ret == ZCONNECTIONLOSS || ret == ZOPERATIONTIMEOUT) 
      {
        LOG(INFO) << "zookeeper create node " << current << " retry for " << zerror(ret);
        continue;
      }

      LOG(ERROR) << "zookeeper create node " << current << " failed: " << zerror(ret);
      return false;
    }
    if (!succ)
    {
      return false;
    }
  }

  return true;
}

bool ZkWrapper::DeleteNode(const std::string &path) 
{
  int ret = zoo_delete(mZkHandle, path.c_str(), -1);
  if(ZOK == ret) 
  {
    return true;
  }
  else 
  {
    return false;
  }
}

bool ZkWrapper::SetNode(const std::string &path, const std::string &str) 
{
  int ret = zoo_set(mZkHandle, path.c_str(), str.c_str(), str.length(), -1);
  if(ZOK == ret) 
  {
    return true;
  }
  else 
  {
    return false;
  }
}

bool ZkWrapper::Validate(const std::string& path)
{
  if (!mZkHandle)
  {
    LOG(ERROR) << "Invalid zookeeper handle.";
    return false;
  }

  if (path.size() == 0 || path[0] != '/')
  {
    LOG(ERROR) << "Invalid zookeeper path " << path;
    return false;
  }
  return true;
}

} // namespace ps

