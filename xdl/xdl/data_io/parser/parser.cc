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

#include <unistd.h>
#include "xdl/data_io/parser/parser.h"
#include "xdl/data_io/parser/parse_pb.h"
#include "xdl/data_io/parser/parse_txt.h"
#include "xdl/data_io/parser/parse_simple.h"
#include "xdl/data_io/parser/parse_v4.h"
#include "xdl/data_io/pool.h"
#include "xdl/core/lib/timer.h"

#include "xdl/core/utils/logging.h"

///TODO return nullptr not END

namespace xdl {
namespace io {

Parser::Parser(ParserType type, const Schema *schema) : schema_(schema) {
  switch(type) {
    case kPB:
      parse_.reset(new ParsePB(schema));
      break;
    case kTxt:
      parse_.reset(new ParseTxt(schema));
      break;
    case kSPB:
      parse_.reset(new ParseSimple(schema));
      break;
    case kV4:
      parse_.reset(new ParseV4(schema));
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

SGroup *Parser::Read2Parse() {
  ssize_t size = -1;
  while (running_) {
again:
    /// try to get size
    size = parse_->GetSize(buf_+begin_, end_-begin_);
    XDL_LOG(DEBUG) << "GetSize=" << size << " begin=" << begin_ << " end=" << end_ << std::endl;
    XDL_CHECK(size != 0 && size < (ssize_t)kBufSize) << "size=" << size;

    XDL_DCHECK(end_-begin_ >= 0);
    if (end_ == begin_ || size > end_-begin_) {
      /// buffer not enough
      size = -1;
    } else if (size > 0) {
      /// read enough
      XDL_CHECK(size <= end_-begin_) << "size=" << size;
      break;
    }

    if (rparam_->begin_ == rparam_->end_) {
      XDL_LOG(DEBUG) << "rparam read over, path=" << rparam_->path_
          << " size=" << rparam_->end_ << " parsed=" << rparam_->parsed_;
      rparam_->parsed_ += size;
      SGroup *sgroup = parse_->Run(nullptr, 0);
      if (sgroup == nullptr) {
        return (SGroup *)END;
      }
      return sgroup;
    }

    /// read more
    uint64_t read_len = (rparam_->begin_ + kReadSize) < (size_t)rparam_->end_ ?
        kReadSize : (rparam_->end_ - rparam_->begin_);

    XDL_LOG(DEBUG) << "try read=" <<  read_len << " rp.begin=" << rparam_->begin_
        << " rp.end=" << rparam_->end_ << std::endl;

    if (end_ + read_len > kBufSize) {
      size_t left = end_ - begin_;
      memmove(buf_, buf_ + begin_, left);
      XDL_LOG(DEBUG) << "move from (" << begin_ << ", " << end_ << ")"
          << " to (0, " << left << ")";
      begin_ = 0;
      end_ = left;
    }

    read_len = rparam_->ant_->Read(buf_ + end_, read_len);
    if (read_len == 0) {
      XDL_LOG(DEBUG) << "read 0 from ant " << running_;
      if (rparam_->end_ == ULONG_MAX) {
        rparam_->end_ = rparam_->begin_;
        XDL_LOG(DEBUG) << "read 0 from zant, end of file, size=" << rparam_->end_;
      } else {
        sleep(1);
      }
      continue;
    }
    XDL_CHECK(read_len > 0 && read_len < kBufSize) << "read len=" << read_len;

    rparam_->begin_ += read_len;
    end_ += read_len;

    XDL_LOG(DEBUG) << "have read=" <<  read_len << " rp.begin=" << rparam_->begin_
        << " rp.end=" << rparam_->end_ << "end=" << end_ << std::endl;
  }
  
  if (!running_) {
    ///force quit
    return (SGroup *)END;
  }

  XDL_CHECK(size > 0) << "size=" << size;

  rparam_->parsed_ += size;
  SGroup *sgroup = parse_->Run(buf_+begin_, size);
  XDL_LOG(DEBUG) << "Run=" << sgroup << " begin=" << begin_ << " end=" << end_ << std::endl;

  XDL_LOG(DEBUG) << "increase begin=" << begin_ << " size=" <<  size
      << "to begin=" << begin_ + size << std::endl;
  begin_ += size;

  if (sgroup == nullptr) {
    goto again;
  }
  return sgroup;
}

SGroup *Parser::Run() {
  //XDL_TIMER_SCOPE(parser_run);
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
