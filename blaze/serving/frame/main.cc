/*
 * \file main.cc 
 * \brief The blaze serving main
 */
#include <string>
#include <iostream>
#include <thread>

#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <evhttp.h>

#include "blaze/api/cpp_api/predictor.h"
#include "serving/frame/process.h"
#include "serving/frame/thread_local.h"
#include "serving/model/model_manager.h"

typedef serving::ThreadLocalStore<serving::ProcessContext> ThreadLocalProcessContextStore;

void callback(evhttp_request * req, void * a) {
  const char* input_data = reinterpret_cast<const char*>(evbuffer_pullup(req->input_buffer, -1));
  size_t input_len = evbuffer_get_length(req->input_buffer);
  std::string input_str(input_data, input_len);

  std::string output_str;
  serving::PredictProcessor predict_processor;
  bool success = predict_processor.Process(input_str, ThreadLocalProcessContextStore::Get(), &output_str);

  struct evbuffer *buf = evbuffer_new();
  if(!buf) {
    std::cerr << "[ERROR] create response buffer fail!" << std::endl;
    return;
  }
  evbuffer_add_printf(buf, output_str.c_str());
  if (success) {
    evhttp_send_reply(req, HTTP_OK, "OK", buf);
  } else {
    std::cout << "error message: " << output_str << std::endl;
    evhttp_send_reply(req, HTTP_NOTFOUND, "NotFound", buf);
  }
  evbuffer_free(buf);
}

int httpserver_bindsocket(int port, int backlog) {
  int r;
  int nfd = socket(AF_INET, SOCK_STREAM, 0);
  if (nfd < 0) {
    std::cerr <<"[ERROR] socket() fail!" << std::endl;
    return -1;
  }

  int one = 1;
  r = setsockopt(nfd, SOL_SOCKET, SO_REUSEADDR, (char *)&one, sizeof(int));
  if (r < 0) {
    std::cerr <<"[ERROR] setsockopt() fail!" << std::endl;
    return -1;
  }

  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  r = bind(nfd, (struct sockaddr*)&addr, sizeof(addr));
  if (r < 0) {
    std::cerr <<"[ERROR] bind() fail!" << std::endl;
    return -1;
  }
  r = listen(nfd, backlog);
  if (r < 0) {
    std::cerr <<"[ERROR] listen() fail!" << std::endl;
    return -1;
  }

  int flags;
  if ((flags = fcntl(nfd, F_GETFL, 0)) < 0
      || fcntl(nfd, F_SETFL, flags | O_NONBLOCK) < 0) {
    std::cerr <<"[ERROR] fcntl() fail!" << std::endl;
    return -1;
  }

  return nfd;
}


int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr <<"Usage: "<<argv[0]<<" <config-file>"<<std::endl;
    return -1;
  }

  if (!serving::ModelManager::Instance()->Init(argv[1])) {
    std::cerr <<"[ERROR] init ModelManager failed！ "<<std::endl;
    return -1;
  }
  blaze::InitScheduler(false, 1000, 100, 32, 4, 2);

  const size_t nthread = 4;
  std::vector<std::thread> threads(nthread);

  int nfd = httpserver_bindsocket(8080, 1024*nthread);
  if (nfd < 0) {
    std::cerr <<"[ERROR] httpserver_bindsocket() failed！ "<<std::endl;
    return -1;
  }

  for (int i = 0 ; i < nthread; ++i) {
    event_base *base = event_init();
    if (base == nullptr) {
      std::cerr << "[ERROR] event_init() fail!" << std::endl;
      return -1;
    }

    evhttp *httpd = evhttp_new(base);
    if (httpd == nullptr) {
      std::cerr << "[ERROR] evhttp_new() fail!" << std::endl;
      return -1;
    }

    int r = evhttp_accept_socket(httpd, nfd);
    if (r != 0) {
      std::cerr << "evhttp_accept_socket() fail!" << std::endl;
      return -1;
    }

    evhttp_set_gencb(httpd, callback, nullptr);
    threads[i] = std::move(std::thread(event_base_dispatch, base));
  }

  for (int i = 0; i < nthread; ++i) {
    threads[i].join();
  }

  return 0;
}
