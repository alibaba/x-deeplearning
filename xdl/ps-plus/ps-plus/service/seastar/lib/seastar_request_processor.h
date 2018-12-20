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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_PROCESSOR_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_PROCESSOR_H_

#include <core/ps_coding/message_processor.hh>
#include <core/ps_coding/fragment.hh>

#include "ps-plus/common/serializer.h"

#include "common.h"
#include "done_closure.h"
#include "seastar_status.h"
#include "server_func_manager.h"
#include "seastar_response_serializer.h"

namespace ps {
class Data;

namespace network {
class SessionContext;
}

namespace service {
namespace seastar {

class SeastarRequestProcessor : public ps::coding::MessageProcessor {
 public:
  ~SeastarRequestProcessor() override {}

  ::seastar::future<> Process(ps::network::SessionContext* sc) override {
    size_t server_func_id;
    std::vector<size_t> serialize_ids;
    ServerFunc server_func;
    SeastarStatus st;
    std::vector<ps::Data*> request_datas;
    ps::serializer::MemGuard mem_guard;
    ParseMeta(&server_func_id, &st, &serialize_ids);
    do {
      if (!st.Success()) break;
      if (0 != ServerFuncManager::GetInstance()->GetServerFunc(
              server_func_id, &server_func)) {
        st = SeastarStatus::ServerFuncNotFound();
        break;
      }

      ps::serializer::Fragment buf;
      buf.base = const_cast<char*>(GetDataBuffer().get());
      buf.size = GetDataBuffer().size();
      request_datas.reserve(serialize_ids.size());        
      size_t offset = 0;
      for (auto id: serialize_ids) {
        ps::Data* data = nullptr;
        size_t len;
        ps::Status deserialize_st = ps::serializer::DeserializeAny<Data>(id, &buf, offset, &data, &len, mem_guard);
        if (!deserialize_st.IsOk()) {
          std::cerr << deserialize_st.ToString() << std::endl;
          st = SeastarStatus::ServerDeserializeFailed();
          break;
        }

        offset += len;
        request_datas.push_back(data);
      }
    } while(0);

    SeastarResponseSerializer* serializer = 
      new SeastarResponseSerializer(this, request_datas, mem_guard, st);
    serializer->SetUserThreadId(GetMessageHeader().mUserThreadId);
    serializer->SetSequence(GetMessageHeader().mSequence);
    server_func(serializer->RequestData(), 
                serializer->MutableResponseData(),
                // server will write packet via queue, 
	        // not directly via write function,
	        // so should pass 'true' param	
                new DoneClosure(sc, serializer, true));
    return ::seastar::make_ready_future<>();
  }

  void ParseMeta(uint64_t* func_id, SeastarStatus* st, std::vector<size_t>* ids) {
    char* cbegin = const_cast<char*>(GetMetaBuffer().begin());
    *func_id = *(reinterpret_cast<uint64_t*>(cbegin));
    cbegin += sizeof(uint64_t);
    int32_t* sbegin = reinterpret_cast<int32_t*>(cbegin);
    *st = SeastarStatus((SeastarStatus::ErrorCode)*sbegin);
    cbegin += sizeof(int32_t);
    size_t* begin = reinterpret_cast<size_t*>(cbegin);
    size_t len = (GetMetaBuffer().size() - sizeof(uint64_t) - sizeof(int32_t)) / sizeof(size_t);
    ids->assign(begin, begin + len);
  }
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(SeastarRequestProcessor, SEASTAR_REQUEST_PROCESSOR_ID);

} // namespace seastar
} // namespace service
} // namespace ps


#endif //PS_SERVICE_SEASTAR_LIB_SEASTAR_REQUEST_PROCESSOR_H_
