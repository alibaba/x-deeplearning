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

#include "event_client_lib.h"
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

using ps::serializer::Fragment;

namespace ps {
class Data;
}

namespace ps {
namespace service {
namespace seastar {

EventReader::EventReader(event_base* base, int fd) {
  fd_ = fd;
  event_ = event_new(base, fd, EV_READ | EV_PERSIST, EventReader::EventReaderCallback, this);
  if (event_ == nullptr) {
    fprintf(stderr, "Event Reader event_ allocate error");
    abort();
  }
  buffer_ptr_ = nullptr;
  buffer_size_ = 0;
  StartReadHeader();
}

EventReader::~EventReader() {
  delete event_;
  delete cur_message_;
}

void EventReader::EventReaderCallback(evutil_socket_t socket, short what, void* arg) {
  EventReader* reader = static_cast<EventReader*>(arg);
  reader->ReadEvent();
}

void EventReader::ReadEvent() {
  read_over_ = false;
  while (!read_over_) {
    int result = Read(cur_ptr_, size_);
    if (result == size_) {
      if (state_ == kReadHeader) {
        StartReadBuffer();
      } else if (state_ == kReadBuffer) {
        ReadDone();
        StartReadHeader();
      } else {
        fprintf(stderr, "reader state_ is not handled %d\n", state_);
        abort();
      }
    } else if (result > 0) {
      cur_ptr_ += result;
      size_ -= result;
    } else if (result < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        break;
      }
      fprintf(stderr, "Disconnected By read errno %d\n", errno);
      Disconnected();
      return;
    } else if (result == 0) {
      fprintf(stderr, "Disconnected By read zero\n");
      Disconnected();
      return;
    }
  }
}

int EventReader::Read(char* buf, int size) {
  if (buffer_size_ > size) {
    memcpy(buf, buffer_ptr_, size);
    buffer_ptr_ += size;
    buffer_size_ -= size;
    return size;
  } else if (buffer_size_ > 0) {
    memcpy(buf, buffer_ptr_, buffer_size_);
    size -= buffer_size_;
    read_over_ = buffer_ptr_ + buffer_size_ < buffer_ + kBufferSize;
    int ret = buffer_size_;
    buffer_size_ = 0;
    return ret;
  }
  if (size > kBufferSize) {
    int ret = read(fd_, buf, size);
    read_over_ = ret < size;
    return ret;
  } else {
    int ret = read(fd_, buffer_, kBufferSize);
    if (ret <= 0) {
      return ret;
    }
    if (size < ret) {
      memcpy(buf, buffer_, size);
      buffer_size_ = ret - size;
      buffer_ptr_ = buffer_ + size;
      return size;
    } else {
      memcpy(buf, buffer_, ret);
      read_over_ = true;
      return ret;
    }
  }
}

void EventReader::StartReadHeader() {
  state_ = kReadHeader;
  cur_message_ = new EventMessage;
  cur_ptr_ = (char*)(void*)&cur_message_->header;
  size_ = sizeof(ps::coding::MessageHeader);
}

void EventReader::StartReadBuffer() {
  state_ = kReadBuffer;
  int size = cur_message_->header.mMetaBufferSize
      + cur_message_->header.mDataBufferSize;
  cur_message_->buffer.reset(new char[size]);
  cur_ptr_ = cur_message_->buffer.get();
  size_ = size;
}

void EventReader::ReadDone() {
  int meta_size = cur_message_->header.mMetaBufferSize;
  int data_size = cur_message_->header.mDataBufferSize;
  char* meta_buf = cur_message_->buffer.get();
  char* data_buf = meta_buf + meta_size;
  SeastarStatus status((SeastarStatus::ErrorCode)(*(int32_t*)(void*)meta_buf));
  ps::serializer::MemGuard mem_guard;
  size_t serializer_size = (meta_size - sizeof(int32_t)) / sizeof(size_t);
  size_t* serializer_ids = (size_t*)(void*)(meta_buf + sizeof(int32_t));
  std::vector<Data*> datas;
  size_t offset = 0;
  ps::serializer::Fragment buf;
  buf.base = data_buf;
  buf.size = data_size;
  for (size_t i = 0 ; i < serializer_size; i++) {
    ps::Data* data = nullptr;
    size_t len;
    Status st = ps::serializer::DeserializeAny<ps::Data>(serializer_ids[i], &buf, offset, &data, &len, mem_guard);
    if (!st.IsOk()) {
      status = SeastarStatus::ClientDeserializeFailed();
      break;
    }

    offset += len;
    datas.push_back(data);
  }
  cur_message_->status = status;
  cur_message_->datas = std::move(datas);
  Process(cur_message_);
  cur_message_ = nullptr;
}

EventWriter::EventWriter(event_base* base, int fd) {
  fd_ = fd;
  event_ = event_new(base, fd, EV_WRITE | EV_PERSIST, EventWriter::EventWriterCallback, this);
  if (event_ == nullptr) {
    fprintf(stderr, "Event Reader event_ allocate error\n");
    abort();
  }
  buffer_ptr_ = nullptr;
  buffer_size_ = 0;
  cur_frag_ = 0;
  size_ = 0;
  stop_ = true;
}

EventWriter::~EventWriter() {
  delete event_;
}

void EventWriter::EventWriterCallback(evutil_socket_t socket, short what, void* arg) {
  EventWriter* writer = static_cast<EventWriter*>(arg);
  writer->WriteEvent();
}

void EventWriter::WriteMessage(EventMessage* message) {
  messages_.push_back(message);
  if (stop_) {
    if (event_add(event_, nullptr) != 0) {
      fprintf(stderr, "Write Event Add Error\n");
      abort();
    }
    stop_ = false;
  }
}

void EventWriter::WriteEvent() {
  write_over_ = false;
  while (true) {
    if (buffer_size_ == 0) {
      if (!FillBuffer()) {
        if (event_del(event_) != 0) {
          fprintf(stderr, "Write Event Del Error\n");
          abort();
        }
        stop_ = true;
        break;
      }
    }
    int ret = write(fd_, buffer_ptr_, buffer_size_);
    if (ret == buffer_size_) {
      buffer_size_ = 0;
      continue;
    } else if (ret > 0) {
      buffer_ptr_ += ret;
      buffer_size_ -= ret;
      break;
    } else if (ret < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      if (errno == EINVAL) {
        continue;
      }
      fprintf(stderr, "Disconnected By read errno %d\n", errno);
      Disconnected();
      return;
    } else if (ret == 0) {
      fprintf(stderr, "write should not return 0\n");
      abort();
    }
  }
}

bool EventWriter::FillBuffer() {
  int buf = 0;
  while (true) {
    if (size_ == 0) {
      if (messages_.empty()) {
        return false;
      }
      if (cur_frag_ == messages_.front()->frags.size()) {
        cur_frag_ = 0;
        DelMessage(messages_.front());
        messages_.pop_front();
        if (messages_.empty()) {
          if (buf == 0) {
            return false;
          }
          buffer_ptr_ = buffer_;
          buffer_size_ = buf;
          return true;
        }
      }
      cur_ptr_ = messages_.front()->frags[cur_frag_].base;
      size_ = messages_.front()->frags[cur_frag_].size;
      cur_frag_++;
    }
    if (buf == 0 && size_ >= kBufferSize) {
      buffer_ptr_ = cur_ptr_;
      buffer_size_ = size_;
      size_ = 0;
      return true;
    }
    if (buf + size_ < kBufferSize) {
      memcpy(buffer_ + buf, cur_ptr_, size_);
      buf += size_;
      size_ = 0;
    } else {
      int x = kBufferSize - buf;
      memcpy(buffer_ + buf, cur_ptr_, x);
      size_ -= x;
      cur_ptr_ += x;
      buf += x;
      buffer_ptr_ = buffer_;
      buffer_size_ = buf;
      return true;
    }
  }
}

void EventWriter::DelMessage(EventMessage* message) {
  if (message->release_datas) {
    for (auto data : message->datas) {
      delete data;
    }
  }
  delete message;
}

EventAsync::EventAsync(event_base *base) {
  event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (event_fd_ < 0) {
    fprintf(stderr, "event fd create error\n");
    abort();
  }
  event_ = event_new(base, event_fd_, EV_READ | EV_PERSIST, EventAsync::EventAsyncCallback, this);
  if (event_ == nullptr) {
    fprintf(stderr, "eventfd event create error\n");
    abort();
  }
  event_sent_ = false;
}

void EventAsync::EventAsyncCallback(evutil_socket_t socket, short what, void* arg) {
  EventAsync* async = static_cast<EventAsync*>(arg);
  async->AsyncEvent();
}

void EventAsync::Async(std::function<void()> func) {
  std::unique_lock<std::mutex> lock(mu_);
  funcs_.push_back(func);
  if (!event_sent_) {
    uint64_t x = 1;
    int ret = write(event_fd_, &x, sizeof(x));
    if (ret <= 0) {
      fprintf(stderr, "eventfd write error, errno=%d\n", errno);
      abort();
    }
    event_sent_ = true;
  }
}

void EventAsync::AsyncEvent() {
  std::vector<std::function<void()>> funcs;
  {
    std::unique_lock<std::mutex> lock(mu_);
    event_sent_ = false;
    uint64_t x;
    int ret = read(event_fd_, &x, sizeof(x));
    if (ret <= 0) {
      fprintf(stderr, "eventfd read error, errno=%d\n", errno);
      abort();
    }
    funcs = funcs_;
    funcs_.clear();
  }
  for (auto func : funcs) {
    func();
  }
}

std::mutex EventHolder::init_mu_;
bool EventHolder::inited_ = false;

EventHolder::EventHolder() {
  {
    std::unique_lock<std::mutex> lock(init_mu_);
    if (!inited_) {
      event_init();
    }
  }
  internal_ = event_base_new();
  async_ = new EventAsync(internal_);
  if (event_add(async_->GetEvent(), nullptr) != 0) {
    fprintf(stderr, "Add Async Event error.\n");
    abort();
  }
  loop_.reset(new std::thread([this]{ event_base_loop(internal_, 0); }));
  loop_thread_id_ = loop_->get_id();
}

EventHolder::~EventHolder() {
  async_->Async([this]{ event_base_loopbreak(internal_); });
  loop_->join();
  delete async_;
  delete internal_;
}

void EventHolder::Async(std::function<void()> func) {
  if (std::this_thread::get_id() == loop_thread_id_) {
    func();
  } else {
    async_->Async(func);
  }
}

EventClientConnection::EventClientConnection(EventHolder* holder, int fd) {
  closed_ = false;
  fd_ = fd;
  holder_ = holder;
  reader_ = new ClientEventReader(this, holder->Internal(), fd);
  writer_ = new ClientEventWriter(this, holder->Internal(), fd);
  holder_->Async([this]{
    if (event_add(reader_->GetEvent(), nullptr) != 0) {
      fprintf(stderr, "Add Event to client error.\n");
      abort();
    }
  });
}

EventClientConnection::~EventClientConnection() {
  Close();
  close_notice_.get_future().wait();
}

bool EventClientConnection::Closed() {
  return closed_;
}

void EventClientConnection::Close() {
  if (!closed_) {
    closed_ = true;
    holder_->Async([this]{
      if (event_del(reader_->GetEvent()) != 0) {
        fprintf(stderr, "Read Event Del In Close Error\n");
        abort();
      }
      if (event_del(writer_->GetEvent()) != 0) {
        fprintf(stderr, "Write Event Del In Close Error\n");
        abort();
      }
      for (auto closure : closures_) {
        CallBackClosure* cb = dynamic_cast<CallBackClosure*>(closure);
        cb->SetStatus(SeastarStatus::NetworkError());
        cb->Run();
      }
      close(fd_);
      close_notice_.set_value(0);
    });
  }
}

void EventClientConnection::Request(
    int32_t func_id, std::vector<Data*> datas,
    Closure* closure, bool delete_request_data) {
  EventWriter::EventMessage* msg = new EventWriter::EventMessage;
  msg->datas = datas;
  msg->release_datas = delete_request_data;
  msg->header.mSequence = reinterpret_cast<uint64_t>(closure);
  msg->header.mProcessorClassId = SEASTAR_REQUEST_PROCESSOR_ID;
  msg->header.mMetaBufferSize = sizeof(uint64_t) + sizeof(int32_t) + sizeof(size_t) * datas.size();
  char* meta = msg->mem.AllocateBuffer(msg->header.mMetaBufferSize);
  *(uint64_t*)(void*)meta = func_id;
  SeastarStatus status;
  msg->frags.push_back(Fragment{.base=(char*)(void*)&msg->header, .size=sizeof(msg->header)});
  msg->frags.push_back(Fragment{.base=meta, .size=msg->header.mMetaBufferSize});
  for (size_t i = 0; i < datas.size(); i++) {
    ps::Status st = ps::serializer::SerializeAny<ps::Data>(datas[i], ((uint64_t*)(void*)(meta + 12)) + i, &msg->frags, msg->mem);
    if (!st.IsOk()) {
      std::cerr << st.ToString() << std::endl;
      status = SeastarStatus::ClientSerializeFailed();
      break;
    }
  }
  *(uint32_t*)(void*)(meta + 8) = status.Code();
  size_t data_size = 0;
  for (size_t i = 2; i < msg->frags.size(); i++) {
    data_size += msg->frags[i].size;
  }
  msg->header.mDataBufferSize = data_size;
  holder_->Async([msg, this, closure]{
    if (!closed_) {
      closures_.insert(closure);
      writer_->WriteMessage(msg); 
    } else {
      CallBackClosure* cb = dynamic_cast<CallBackClosure*>(closure);
      cb->SetStatus(SeastarStatus::NetworkError());
      cb->Run();
    }
  });
}

void EventClientConnection::ClientEventReader::Process(EventMessage* message) {
  Closure* closure = reinterpret_cast<Closure*>(message->header.mSequence);
  CallBackClosure* cb = dynamic_cast<CallBackClosure*>(closure);
  ps::serializer::MemGuard mem_guard;
  cb->SetResponseData(message->datas);
  cb->SetMemGuard(mem_guard);
  cb->SetStatus(message->status);
  cb->Run();
  delete message;
  conn_->closures_.erase(closure);
}

void EventClientConnection::ClientEventReader::Disconnected() {
  conn_->Close();
}

void EventClientConnection::ClientEventWriter::Disconnected() {
  conn_->Close();
}

bool EventClientLib::Start() {
  for (auto&& event : events_) {
    event.reset(new EventHolder);
  }
}

void EventClientLib::Stop() {
  for (auto&& conn : conns_) {
    conn.reset(nullptr);
  }
  for (auto&& event : events_) {
    event.reset(nullptr);
  }
}

void EventClientLib::Request(
    int32_t server_id, 
    int32_t func_id,
    const std::vector<ps::Data*>& request_datas,
    Closure* closure,
    bool delete_request_data) {
  conns_[server_id]->Request(func_id, request_datas, closure, delete_request_data);
}

inline int setopt(int fd, int type, int k, int v) {
  return setsockopt(fd, type, k, &v, sizeof(v));
}

bool EventClientLib::Connect(
    const int32_t server_id, 
    const std::string& server_addr) {
  std::string host_str, port_str;
  size_t pos = server_addr.find(':');
  if (pos == std::string::npos) {
    return false;
  }
  host_str = server_addr.substr(0, pos);
  port_str = server_addr.substr(pos + 1);
  hostent* host = gethostbyname(host_str.c_str());
  if (host == nullptr) {
    return false;
  }
  sockaddr_in addr;
  bzero((char *) &addr, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(atoi(port_str.c_str()));
  addr.sin_addr = *((struct in_addr *)host->h_addr);
  int sd;
  sd = ::socket(PF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (sd <= 0) {
    return false;
  }
  if (setopt(sd, IPPROTO_TCP, TCP_NODELAY, 1)) {
    std::cerr << "SET OPT TCP_NODELAY ERROR\n";
    return false;
  }
  if (setopt(sd, SOL_SOCKET, SO_KEEPALIVE, 1)) {
    std::cerr << "SET OPT SO_KEEPALIVE ERROR\n";
    return false;
  }
  if (setopt(sd, IPPROTO_TCP, TCP_KEEPIDLE, 300)) {
    std::cerr << "SET OPT TCP_KEEPALIVE ERROR\n";
    return false;
  }
  if (setopt(sd, IPPROTO_TCP, TCP_KEEPINTVL, 10)) {
    std::cerr << "SET OPT TCP_KEEPINTVL ERROR\n";
    return false;
  }
  if (setopt(sd, IPPROTO_TCP, TCP_KEEPCNT, 6)) {
    std::cerr << "SET OPT TCP_KEEPINTVL ERROR\n";
    return false;
  }
  if (connect(sd, (const struct sockaddr*)&addr, sizeof(addr))) {
    if (errno != EINPROGRESS) {
      return false;
    }
  }
  conns_.resize(std::max(conns_.size(), (size_t)server_id + 1));
  conns_[server_id].reset(new EventClientConnection(events_[server_id % events_.size()].get(), sd));
  return true;
}

int EventClientLib::CheckServer() {
  std::this_thread::sleep_for(std::chrono::seconds(1));
  for (size_t i = 0; i < conns_.size(); i++) {
    if (conns_[i]->Closed()) {
      return i + 1;
    }
  }
  return 0;
}

} // namespace seastar
} // namespace service
} // namespace ps
