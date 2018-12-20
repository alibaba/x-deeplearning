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

#include "xdl/data_io/parser/parser.h"

#include <unistd.h>

#include "xdl/data_io/parser/parse_pb.h"
#include "xdl/data_io/parser/parse_txt.h"
#include "xdl/data_io/parser/parse_v4.h"
#include "xdl/data_io/parser/parse_simple.h"
#include "xdl/data_io/pool.h"

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

Parser::Parser(ParserType type, const Schema *schema) : schema_(schema) {
  switch(type) {
    case kPB:
      parse_ = new ParsePB(schema);
      break;
    case kTxt:
      parse_ = new ParseTxt(schema);
      break;
    case kV4:
      parse_ = new ParseV4(schema);
      break;
    case kSPB:
      parse_ = new ParseSimple(schema);
      break;
    default:
      XDL_LOG(FATAL) << "invalid parser_type=" << type;
  }
}

bool Parser::InitMeta(const std::string &meta) {
  return parse_->InitMeta(meta);
}

bool Parser::Init(ReadParam *rparam) {
  begin_ = 0;
  end_ = 0;
  rparam_ = rparam;
  return true;
}

SGroup *Parser::Run1() {
  uint32_t len = ((uint32_t *)(buf_+begin_))[0];
  XDL_CHECK(len < kBufSize && len < kReadSize);
  if (len + sizeof(uint32_t) + begin_ > (size_t)end_) {
    if (rparam_->begin_ == rparam_->end_) {
      return (SGroup *)END;
    }
    /// read more
    uint64_t read_len = (rparam_->begin_ + kReadSize) < (size_t)rparam_->end_ ?
        kReadSize : (rparam_->end_ - rparam_->begin_);

    XDL_DLOG(DEBUG) << "read more, len=" << len << " read=" <<  read_len
        << " begin=" << rparam_->begin_ << " end=" << rparam_->end_ << std::endl;

    if (end_ + read_len > kBufSize) {
      size_t left = end_ - begin_;
      memmove(buf_, buf_+begin_, left);
      begin_ = 0;
      end_ = left;
    }

    read_len = rparam_->ant_->Read(buf_+end_, read_len);
    rparam_->begin_ += read_len;
    end_ += read_len;

    len = ((uint32_t *)(buf_+begin_))[0];
  }

  SGroup *sgroup = SGroupPool::Get()->Acquire();
  SampleGroup *sg = sgroup->New();
  XDL_CHECK(sg->ParseFromArray(buf_ + begin_ + sizeof(uint32_t), len))
      << "parse sample group failed, len=" << len;

  sgroup->Reset();

  begin_ += sizeof(uint32_t) + len;

  return sgroup;
}

SGroup *Parser::Read2Parse() {
  ssize_t size = -1;
  while (running_) {
    size = parse_->GetSize(buf_+begin_, end_-begin_);
    XDL_DLOG(DEBUG) << "GetSize=" << size << " begin=" << begin_ << " end=" << end_ << std::endl;

    XDL_CHECK(size != 0 && size < (ssize_t)kBufSize) << "size=" << size;
    if (size > 0 && size <= end_-begin_) {
      /// read enough
      break;
    }

    if (rparam_->begin_ == rparam_->end_) {
      XDL_LOG(DEBUG) << "rparam read over, path=" << rparam_->path_
          << " size=" << rparam_->end_;
      SGroup *sgroup = parse_->Run(nullptr, 0);
      if (sgroup == nullptr) {
        return (SGroup *)END;
      }
      return sgroup;
    }

    /// read more
    uint64_t read_len = (rparam_->begin_ + kReadSize) < (size_t)rparam_->end_ ?
        kReadSize : (rparam_->end_ - rparam_->begin_);

    XDL_DLOG(DEBUG) << "try read=" <<  read_len << " rp.begin=" << rparam_->begin_
        << " rp.end=" << rparam_->end_ << std::endl;

    if (end_ + read_len > kBufSize) {
      size_t left = end_ - begin_;
      memmove(buf_, buf_ + begin_, left);
      XDL_DLOG(DEBUG) << "move from (" << begin_ << ", " << end_ << ")"
          << " to (0, " << left << ")";
      begin_ = 0;
      end_ = left;
    }

    read_len = rparam_->ant_->Read(buf_ + end_, read_len);
    if (read_len == 0) {
      XDL_DLOG(DEBUG) << "read 0 from ant " << running_;
      sleep(1);
      continue;
    }

    rparam_->begin_ += read_len;
    end_ += read_len;

    XDL_DLOG(DEBUG) << "have read=" <<  read_len << " rp.begin=" << rparam_->begin_
        << " rp.end=" << rparam_->end_ << "end=" << end_ << std::endl;
  }
  
  if (!running_) {
    ///force quit
    return (SGroup *)END;
  }

  SGroup *sgroup = parse_->Run(buf_+begin_, size);
  XDL_DLOG(DEBUG) << "Run=" << sgroup << " begin=" << begin_ << " end=" << end_ << std::endl;

  XDL_DLOG(DEBUG) << "increase begin=" << begin_ << " size=" <<  size
      << "to begin=" << begin_ + size << std::endl;
  begin_ += size;

  return sgroup;
}

SGroup *Parser::Run() {
  SGroup *sgroup = nullptr;
  do {
    sgroup = Read2Parse();
  } while (sgroup == nullptr);
  return sgroup;
}

bool Parser::Shutdown() {
  running_ = false;
  return true;
}

}  // namespace io
}  // namespace xdl
