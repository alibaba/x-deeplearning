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

#ifndef PS_SERVICE_SEASTAR_LIB_EVENT_CLIENT_LIB_H_
#define PS_SERVICE_SEASTAR_LIB_EVENT_CLIENT_LIB_H_

#include <thread>
#include <memory>
#include <tuple>
#include <vector>
#include <future>
#include <service/client_network_context.hh>
#include <event.h>
#include "ps-plus/common/rd_lock.h"
#include "ps-plus/service/seastar/lib/callback_closure.h"

namespace ps {
class Data;
}

namespace ps {
namespace service {
namespace seastar {

class EventReader {
 public:
  struct EventMessage {
    ps::coding::MessageHeader header;
    std::unique_ptr<char[]> buffer;
    SeastarStatus status;
    std::vector<Data*> datas;
  };
  EventReader(event_base* base, int fd);
  ~EventReader();
  static void EventReaderCallback(evutil_socket_t socket, short what, void* arg);
  event* GetEvent() { return event_; }
  virtual void Process(EventMessage* message) = 0;
  virtual void Disconnected() = 0;
 private:
  void ReadEvent();
  int Read(char* buf, int size);
  void StartReadHeader();
  void StartReadBuffer();
  void ReadDone();

  static constexpr int kBufferSize = 8192;  // At least read 8KB per read syscall.

  enum State {
    kReadHeader,
    kReadBuffer
  };

  int fd_;
  event* event_;

  char buffer_[kBufferSize];
  char* buffer_ptr_;
  int buffer_size_;

  EventMessage* cur_message_;
  bool read_over_;
  char* cur_ptr_;
  int size_;
  State state_;
};

class EventWriter {
 public:
  struct EventMessage {
    bool release_datas;
    ps::coding::MessageHeader header;
    std::vector<Data*> datas;
    ps::serializer::MemGuard mem;
    std::vector<ps::serializer::Fragment> frags;
  };
  EventWriter(event_base* base, int fd);
  ~EventWriter();
  static void EventWriterCallback(evutil_socket_t socket, short what, void* arg);
  event* GetEvent() { return event_; }
  void WriteMessage(EventMessage* message);
  virtual void Disconnected() = 0;
 private:
  void WriteEvent();
  bool FillBuffer();
  void DelMessage(EventMessage* message);

  static constexpr int kBufferSize = 8192;  // At least write 8KB per read syscall.

  int fd_;
  event* event_;
  bool stop_;

  bool write_over_;
  char buffer_[kBufferSize];

  char* buffer_ptr_;
  int buffer_size_;

  std::deque<EventMessage*> messages_;
  int cur_frag_;
  char* cur_ptr_;
  int size_;
};

class EventAsync {
 public:
  EventAsync(event_base *base);
  static void EventAsyncCallback(evutil_socket_t socket, short what, void* arg);
  event* GetEvent() { return event_; }
  void Async(std::function<void()> func);
 private:
  void AsyncEvent();
  std::mutex mu_;
  int event_fd_;
  event* event_;
  bool event_sent_;
  std::vector<std::function<void()>> funcs_;
};

class EventHolder {
 public:
  EventHolder();
  ~EventHolder();
  void Async(std::function<void()> func);
  event_base* Internal() { return internal_; }
 private:
  static std::mutex init_mu_;
  static bool inited_;
  EventAsync* async_;
  event_base* internal_;
  std::unique_ptr<std::thread> loop_;
  std::thread::id loop_thread_id_;
};

class EventClientConnection {
 public:
  EventClientConnection(EventHolder* holder, int fd);
  ~EventClientConnection();
  bool Closed();
  void Close();
  void Request(int32_t func_id, std::vector<Data*> datas,
               Closure* closure, bool delete_request_data);
 private:
  class ClientEventReader : public EventReader {
   public:
    ClientEventReader(EventClientConnection* conn, event_base* base, int fd) : EventReader(base, fd), conn_(conn) {}
    void Process(EventMessage* message) override;
    void Disconnected() override;
   private:
    EventClientConnection* conn_;
  };
  class ClientEventWriter : public EventWriter {
   public:
    ClientEventWriter(EventClientConnection* conn, event_base* base, int fd) : EventWriter(base, fd), conn_(conn) {}
    void Disconnected() override;
   private:
    EventClientConnection* conn_;
  };
  std::atomic<bool> closed_;
  std::promise<int> close_notice_;
  int fd_;
  EventHolder* holder_;
  ClientEventReader* reader_;
  EventWriter* writer_;
  std::set<Closure*> closures_;
};

class EventClientLib {
 public:
  using ServerAddr = std::tuple<int64_t, std::string>;
  EventClientLib(const std::vector<ServerAddr>& server_addrs,
                 int user_thread_num, 
                 int core_num,
                 size_t timeout = 0) {
    events_.resize(core_num);
  }

  bool Start();
  void Stop();
  void Request(int32_t server_id, 
               int32_t func_id,
               const std::vector<ps::Data*>& request_datas,
               Closure* closure,
               bool delete_request_data = true);

  bool Connect(const int32_t server_id, 
               const std::string& server_addr);

  int CheckServer();
 private:
  std::vector<std::unique_ptr<EventHolder>> events_;
  std::vector<std::unique_ptr<EventClientConnection>> conns_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_EVENT_CLIENT_LIB_H_
