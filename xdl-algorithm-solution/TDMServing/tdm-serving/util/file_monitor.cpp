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

#include "util/file_monitor.h"

#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <libgen.h>
#include <pthread.h>
#include <sys/inotify.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

#include <list>
#include <map>
#include <memory>
#include <algorithm>

#include "util/log.h"

namespace tdm_serving {
namespace util {

enum FileStat {
  S_NOT_EXIST,
  S_EXIST,
};

struct MonitorCookie {
  MonitorCookie(const std::string& to_watch_file,
                WatchAction action,
                void* args)
      : watch_file(to_watch_file), action(action), args(args), mtime(0),
        stat(S_NOT_EXIST), wd(-1) {
  }
  ~MonitorCookie() {
  }
  std::string watch_file;
  WatchAction action;
  void* args;
  time_t mtime;
  FileStat stat;
  int wd;
};

#define TOKEN_PASTE(x, y) x##y
#define CAT(x, y) TOKEN_PASTE(x, y)
#define SCOPE_LOCK(mutex)\
  pthread_mutex_lock(mutex);\
  std::shared_ptr<pthread_mutex_t> \
      CAT(lock_, __LINE__) (mutex, pthread_mutex_unlock);

typedef std::list<MonitorCookie*> MonitorList;
typedef MonitorList::iterator MonitorListIterator;
typedef std::map<std::string, MonitorList> MonitorListMap;
typedef MonitorListMap::iterator MonitorListMapIterator;

static MonitorListMap g_watched_file_map;
static pthread_t g_watched_thread = 0;
static pthread_mutex_t g_monitor_mutex;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static std::map<int, MonitorCookie*> g_wd_cookie_map;
static int g_inotify_fd = -1;

static void FileMonitorInit();
static void FileMonitorDestory();
static void* EventLoop(void* arg);

static void AddToInotify(MonitorCookie* cookie) {
  std::string tmp_watch_file(cookie->watch_file.c_str());
  inotify_add_watch(g_inotify_fd, dirname(&tmp_watch_file[0]), IN_CREATE);
  int wd = inotify_add_watch(g_inotify_fd, cookie->watch_file.c_str(),
                             IN_MODIFY | IN_DELETE_SELF | IN_MOVE_SELF);
  if (wd == -1) {
    fprintf(stderr, "inotify_add_watch error:%s\n",
        cookie->watch_file.c_str());
    return;
  }
  cookie->wd = wd;
  g_wd_cookie_map.insert(std::make_pair(cookie->wd, cookie));
}

static void RemoveFromInotify(MonitorCookie* cookie) {
  std::map<int, MonitorCookie*>::iterator it =
      g_wd_cookie_map.find(cookie->wd);
  if (it != g_wd_cookie_map.end() && cookie->wd != -1) {
    inotify_rm_watch(g_inotify_fd, cookie->wd);
    g_wd_cookie_map.erase(it);
  }
}

static void Process(const std::string& file, MonitorCookie* cookie) {
  struct ::stat st;
  int ret = stat(file.c_str(), &st);
  if (ret == -1) {
    if (errno == ENOENT && cookie->stat == S_EXIST) {
      cookie->action(file, WE_DELETE, cookie->args);
      cookie->stat = S_NOT_EXIST;
      RemoveFromInotify(cookie);
      return;
    }
    return;
  }
  if (cookie->stat == S_NOT_EXIST) {
    cookie->action(file, WE_CREATE, cookie->args);
    cookie->stat = S_EXIST;
    cookie->mtime = st.st_mtime;
    AddToInotify(cookie);
  }
  if (cookie->mtime != st.st_mtime) {
    cookie->action(file, WE_MODIFY, cookie->args);
    cookie->mtime = st.st_mtime;
  }
}

static bool g_is_exited = false;

static void FinishMonitor() {
  g_is_exited = true;
  if (g_watched_thread != 0) {
    pthread_join(g_watched_thread, NULL);
  }
  FileMonitorDestory();
}

void WaitInotifyEventAndProcess() {
  struct timeval tv = {tv_sec: 3, tv_usec: 0};
  fd_set read_set;
  FD_ZERO(&read_set);
  FD_SET(g_inotify_fd, &read_set);
  int ret = select(g_inotify_fd + 1, &read_set, NULL, NULL, &tv);
  if (ret == -1) {
    fprintf(stderr, "select error:%s\n", strerror(errno));
    return;
  }
  if (ret < 0) {
    return;
  }
  std::string buffer;
  char buff[256] = "";
  int len = 0;
  while ((len = read(g_inotify_fd, buff, sizeof(buff))) > 0) {
    buffer.append(buff, len);
    if (len < static_cast<int>(sizeof(buff))) {
      break;
    }
  }
  if (buffer.size() > 0) {
    char* ptr = &buffer[0];
    char* eptr = ptr + buffer.size();
    while (ptr < eptr) {
      struct inotify_event* event =
          reinterpret_cast<struct inotify_event*>(ptr);
      std::map<int, MonitorCookie*>::iterator it =
          g_wd_cookie_map.find(event->wd);
      if (it != g_wd_cookie_map.end()) {
        MonitorCookie* cookie = it->second;
        Process(cookie->watch_file, cookie);
      }
      ptr += sizeof(*event) + event->len;
    }
  }
}

static void* EventLoop(void* /*arg*/) {
  fprintf(stdout, "FileMonitor begin EventLoop\n");
  while (!g_is_exited) {
    try {
      {
        MonitorListMapIterator mit = g_watched_file_map.begin();
        for (; mit != g_watched_file_map.end(); ++mit) {
          for (MonitorListIterator it = mit->second.begin();
               it != mit->second.end(); it++) {
            Process(mit->first, *it);
          }
        }
      }
      usleep(1000);
    } catch (...) { }
  }

  fprintf(stdout, "FileMonitor end EventLoop\n");
  return NULL;
}

static void FileMonitorInit() {
  int ec = pthread_mutex_init(&g_monitor_mutex, NULL);
  assert(ec == 0);
  ec = pthread_create(&g_watched_thread, NULL, EventLoop, NULL);
  assert(ec == 0);
  g_inotify_fd = inotify_init();
  assert(g_inotify_fd != -1);
  int flag = fcntl(g_inotify_fd, F_GETFL);
  assert(flag != -1);
  fcntl(g_inotify_fd, F_SETFL, flag | O_NONBLOCK);
  atexit(FinishMonitor);
}

static void FileMonitorDestory() {
  pthread_mutex_destroy(&g_monitor_mutex);
  g_watched_thread = -1;
  close(g_inotify_fd);
}

MonitorCookie* FileMonitor::Watch(const std::string& to_watch_file,
                                  WatchAction action,
                                  void* args) {
  if (to_watch_file.empty()) {
    fprintf(stdout, "to_watch_file is null.\n");
    return NULL;
  }
  if (g_is_exited) {
    return NULL;
  }
  pthread_once(&once, FileMonitorInit);
  SCOPE_LOCK(&g_monitor_mutex);

  MonitorCookie* cookie = new MonitorCookie(to_watch_file, action, args);
  struct stat st;
  int ret = stat(to_watch_file.c_str(), &st);
  if (ret == 0) {
    cookie->mtime = st.st_mtime;
    cookie->stat = S_EXIST;
    AddToInotify(cookie);
  }
  MonitorListMapIterator mit = g_watched_file_map.find(cookie->watch_file);
  if (mit == g_watched_file_map.end()) {
    mit = g_watched_file_map.insert(g_watched_file_map.end(),
                                    std::make_pair(cookie->watch_file,
                                                   MonitorList()));
  }
  mit->second.insert(mit->second.end(), cookie);
  return cookie;
}

void FileMonitor::UnWatch(const std::string& to_watch_file) {
  if (g_is_exited) {
    return;
  }
  SCOPE_LOCK(&g_monitor_mutex);
  MonitorListMapIterator mit = g_watched_file_map.find(to_watch_file);
  if (mit != g_watched_file_map.end()) {
    MonitorListIterator lit = mit->second.begin();
    for (; lit != mit->second.end(); lit++) {
      RemoveFromInotify(*lit);
      delete *lit;
    }
    g_watched_file_map.erase(mit);
  }
}

void FileMonitor::UnWatch(MonitorCookie* cookie) {
  if (g_is_exited) {
    return;
  }
  SCOPE_LOCK(&g_monitor_mutex);
  MonitorListMapIterator mit = g_watched_file_map.find(cookie->watch_file);
  if (mit != g_watched_file_map.end()) {
    MonitorListIterator lit = std::find(mit->second.begin(),
                                        mit->second.end(), cookie);
    if (lit != mit->second.end()) {
      RemoveFromInotify(*lit);
      delete *lit;
      mit->second.erase(lit);
    }
    if (mit->second.size() == 0) {
      g_watched_file_map.erase(mit);
    }
  }
}

}  // namespace util
}  // namespace tdm_serving

