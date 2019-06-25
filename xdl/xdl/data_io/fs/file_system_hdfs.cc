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

#include "xdl/data_io/fs/file_system_hdfs.h"

#include <memory>
#include <omp.h>
#include <stdlib.h>

#include "xdl/core/utils/logging.h"

namespace {
const char* kNameServiceKey = "NAME_SERVICE";
}  // namespace 

namespace xdl {
namespace io {

/// Hdfs IO ant
class IOAntHdfs: public IOAnt {
 public:
  IOAntHdfs(LibHDFS* hdfs, hdfsFS fs, hdfsFile fd) 
  : fs_(fs)
  , fd_(fd) 
  , hdfs_(hdfs) {}
  ~IOAntHdfs() { 
    hdfs_->hdfsHSync(fs_, fd_);
    XDL_CHECK(hdfs_->hdfsCloseFile(fs_, fd_) == 0); 
  }

  /*!\brief read data */
  virtual ssize_t Read(char *data, size_t len) override {
    size_t nleft = len;
    while (nleft != 0) {
      tSize ret = hdfs_->hdfsRead(fs_, fd_, data, nleft);
      if (ret > 0) {
        size_t n = static_cast<size_t>(ret);
        nleft -= n; data += n;
      } else if (ret == 0) {
        break;
      } else {
        int errsv = errno;
        if (errno == EINTR) continue;
        XDL_LOG(FATAL) << "HDFSStream.hdfsRead Error:" << strerror(errsv);
        return -1;
      }
    }
    return len - nleft;
  }

  /*!\brief write data */
  virtual ssize_t Write(const char *data, size_t len) override {
    size_t flen = len;
    while (len != 0) {
      tSize nwrite = hdfs_->hdfsWrite(fs_, fd_, data, len);
      if (nwrite == -1) {
        int errsv = errno;
        XDL_LOG(FATAL) << "HDFSStream.hdfsWrite Error:" << strerror(errsv);
        return -1;
      }
      size_t sz = static_cast<size_t>(nwrite);
      data += sz; len -= sz;
    }
    return flen;
  }

  /*!\brief seek to offset */
  virtual off_t Seek(off_t offset) override {
    XDL_CHECK(hdfs_->hdfsSeek(fs_, fd_, offset) == 0);
    return offset;
  }

 protected:
  /*! hdfs io */
  hdfsFS fs_;
  /*! hdfs file */
  hdfsFile fd_;

  LibHDFS* hdfs_;
};

IOAnt *FileSystemHdfs::GetAnt(const char *path, char mode) {
  hdfsFile fd = mode == 'r' ? 
                hdfs_->hdfsOpenFile(fs_, path, O_RDONLY, 0, 0, 0) :
                hdfs_->hdfsOpenFile(fs_, path, O_WRONLY|O_CREAT, 0, 0, 0);
  XDL_CHECK(fd != nullptr) << "Open " << path << " failed";
  return new IOAntHdfs(hdfs_, fs_, fd);
}

bool FileSystemHdfs::IsReg(const char *path) {
  hdfsFileInfo *info = hdfs_->hdfsGetPathInfo(fs_, path);
  if (info == nullptr) {
    return false;
  }
  bool ret = (info->mKind == kObjectKindFile);

  hdfs_->hdfsFreeFileInfo(info, 1);
  return ret;
}

bool FileSystemHdfs::IsDir(const char *path) {
  hdfsFileInfo *info = hdfs_->hdfsGetPathInfo(fs_, path);
  if (info == nullptr) {
    return false;
  }
  bool ret = (info->mKind == kObjectKindDirectory);

  hdfs_->hdfsFreeFileInfo(info, 1);
  return ret;
}

std::vector<std::string> FileSystemHdfs::Dir(const char *path) {
  std::vector<std::string> paths;

  int count;
  hdfsFileInfo *info= hdfs_->hdfsListDirectory(fs_, path, &count);
  XDL_CHECK(info != nullptr) << "can't open dir " << path;
  for(int i = 0; i < count; ++i) {
    paths.push_back(info[i].mName);
  } 

  hdfs_->hdfsFreeFileInfo(info, count);

  return paths;
}

void *FileSystemHdfs::Open(const char *path, const char *mode) {
  int o = O_RDONLY;
  if (mode[0] == 'w') {
    o |= O_WRONLY;
  }
  hdfsFile fd = hdfs_->hdfsOpenFile(fs_, path, o, 0, 0, 0);
  XDL_CHECK(fd != nullptr) << "can't open " << path;
  return fd;
}

size_t FileSystemHdfs::Size(const char *path) {
  hdfsFileInfo *info = hdfs_->hdfsGetPathInfo(fs_, path);
  XDL_CHECK(info != nullptr) << "can't stat " << path;
  size_t size = info->mSize;

  hdfs_->hdfsFreeFileInfo(info, 1);
  return size;
}

std::string FileSystemHdfs::Path2Node(const std::string &path) {
  const std::string tok = "hdfs://";
  const std::string tok1 = "/";
  size_t pos = path.find(tok);
  if (pos == std::string::npos) {
    return "";
  }
  XDL_CHECK(pos == 0) << "path must begin with 'hdfs://', path=" << path;
  pos = path.find(tok1, tok.size());
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(0, pos);
}

std::string FileSystemHdfs::Read(const std::string &path) {
  auto namenode = Path2Node(path);
  /*
  std::shared_ptr<FileSystemHdfs> fs(this);
  if (!namenode.empty() && strncmp(namenode.c_str(), namenode_.c_str(), namenode.size())) {
    XDL_LOG(DEBUG) << "read from a new fs, main=" << namenode_ << "new=" << namenode;
    fs.reset(new FileSystemHdfs(namenode.c_str())); 
  }
  */

  std::unique_ptr<IOAnt> io(this->GetAnt(path.c_str()));
  size_t len = this->Size(path.c_str());
  XDL_CHECK(len < 1024*1024);
  std::string ret;
  ret.resize(len);
  XDL_CHECK(io->Read(&ret[0], len) == len) << "read failed, path=" << path;
  return ret;
}
  
bool FileSystemHdfs::Write(const std::string &path, const std::string &content) {
  auto namenode = Path2Node(path);
  /*
  std::shared_ptr<FileSystemHdfs> fs(this);
  if (!namenode.empty() && strncmp(namenode.c_str(), namenode_.c_str(), namenode.size())) {
    XDL_LOG(DEBUG) << "read from a new fs, main=" << namenode_ << "new=" << namenode;
    fs.reset(new FileSystemHdfs(namenode.c_str())); 
  }
  */

  std::unique_ptr<IOAnt> io(this->GetAnt(path.c_str(), 'w'));
  size_t len = content.size();
  XDL_CHECK(len == io->Write(content.data(), len)) << "write failed, path=" << path;
  return true;
}

/// FileSystemHdfs Implementation
FileSystemHdfs::~FileSystemHdfs() { }

FileSystem *FileSystemHdfs::Get(const char* namenode) {
    static std::map<std::string, std::shared_ptr<FileSystemHdfs> > inst;
    std::string nn = namenode;
    auto iter = inst.find(nn);
    if(iter != inst.end()){
        return iter->second.get();
    }
    std::shared_ptr<FileSystemHdfs> nf(new FileSystemHdfs(namenode));
    inst[nn] = nf;
    return nf.get();
}

FileSystemHdfs::FileSystemHdfs(const char* namenode) : namenode_(namenode) {
  LibHDFS::Load(&hdfs_);    
  if (namenode_.empty()) {
    namenode_.append(getenv(kNameServiceKey));
    XDL_LOG(DEBUG) << "nameservice=" << namenode_;
  }
  fs_ = hdfs_->hdfsConnect(namenode_.c_str(), 0);
  XDL_CHECK(fs_ != nullptr) << "Failed to load HDFS-configuration:" << namenode;
}

FileSystem* GetHdfsFileSystem(const std::string& name) {
  size_t pos = name.find("://");
  XDL_CHECK(pos != std::string::npos) << "file name error";
  XDL_CHECK(name.substr(0, pos) == "hdfs") << "not hdfs file";
  pos = name.find("/", pos + 3);
  XDL_CHECK(pos != std::string::npos) << "file name error";
  std::string namenode = name.substr(0, pos + 1);
  return FileSystemHdfs::Get(namenode.c_str());
}

void HdfsWrite(const std::string& name, const std::string& content) {
  FileSystem* fs = GetHdfsFileSystem(name);
  std::unique_ptr<IOAnt> io(fs->GetAnt(name.c_str(), 'w'));
  size_t len = content.size();
  XDL_CHECK(len == io->Write(content.data(), len)) << "write failed";
}

std::string HdfsRead(const std::string& name) {
  FileSystem* fs = GetHdfsFileSystem(name);
  size_t len = fs->Size(name.c_str());
  std::unique_ptr<IOAnt> io(fs->GetAnt(name.c_str()));
  std::string ret;
  ret.resize(len);
  XDL_CHECK(io->Read(&ret[0], len) == len) << "read failed";
  return ret;
}

}  // namespace io
}  // namespace xdl
