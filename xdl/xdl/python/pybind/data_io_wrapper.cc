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

#include "xdl/python/pybind/data_io_wrapper.h"

#include <future>

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "xdl/data_io/data_io.h"
#include "xdl/data_io/op/debug_op.h"
#include "xdl/data_io/fs/file_system_hdfs.h"

namespace xdl {
namespace python_lib {

using namespace xdl::io;

io::Operator *GetIOP(const std::string &key) {
  return IOPRegistry::Get()->GetOP(key);
}

void DataIOPybind(pybind11::module& m) {
  pybind11::enum_<FSType>(m, "fs")
    .value("local", kLocal)
    .value("hdfs", kHdfs)
    .value("kafka", kKafka);

  pybind11::enum_<ParserType>(m, "parsers")
    .value("pb", kPB)
    .value("txt", kTxt)
    .value("spb", kSPB)
    .value("v4", kV4);  

  pybind11::enum_<FeatureType>(m, "features")
    .value("sparse", kSparse)
    .value("dense", kDense);

  pybind11::enum_<ZType>(m, "ztype")
    .value("raw", kRaw)
    .value("zlib", kZLib);

  pybind11::class_<FileSystem>(m, "FileSystem")
    .def("get_ant", &FileSystem::GetAnt)
    .def("is_dir", &FileSystem::IsDir)
    .def("is_reg", &FileSystem::IsReg)
    .def("dir", &FileSystem::Dir)
    .def("size", &FileSystem::Size)
    .def("read", &FileSystem::Read)
    .def("write", &FileSystem::Write);

  pybind11::class_<IOAnt>(m, "IOAnt")
    .def("read", &IOAnt::Read)
    .def("write", &IOAnt::Write)
    .def("seek", &IOAnt::Seek);

  pybind11::class_<Block>(m, "Block");

  pybind11::class_<Batch>(m, "Batch")
    .def("blocks", &Batch::blocks)
    .def("get", &Batch::Get);

  pybind11::class_<DataIO>(m, "DataIO")
    .def(pybind11::init<const std::string &, ParserType,
         FSType, const std::string &, size_t, bool >(), "creat a dataio",
         pybind11::arg("name"),
         pybind11::arg("file_type")=kPB,
         pybind11::arg("fs_type")=kLocal,
         pybind11::arg("namenode")="",
         pybind11::arg("worker_id")=0,
         pybind11::arg("global_schedule")=false)
    .def("destroy", &DataIO::Destroy)
    .def("startup", &DataIO::Startup)
    .def("restart", &DataIO::Restart)
    .def("shutdown", &DataIO::Shutdown, "shutdown",
         pybind11::arg("force")=false)
    .def("add_path", &DataIO::AddPath)
    .def("set_meta", &DataIO::SetMeta)
    .def("set_meta_data", &DataIO::SetMetaData)
    .def("add_op", &DataIO::AddOp)
    .def("batch_size", &DataIO::SetBatchSize)
    .def("shuffle", &DataIO::SetShuffle, "set shuffle", pybind11::arg("shuffle")=true)
    .def("pad", &DataIO::SetPadding, "set padding", pybind11::arg("pad")=true)
    .def("epochs", &DataIO::SetEpochs)
    .def("z", &DataIO::SetZType, "set compression type", pybind11::arg("type")=kZLib)
    .def("label_count", &DataIO::SetLabelCount)
    .def("split_group", &DataIO::SetSplitGroup)
    .def("unique_ids", &DataIO::SetUniqueIds)
    .def("finish_delay", &DataIO::SetFinishDelay)
    .def("keep_sample", &DataIO::SetKeepSGroup)
    .def("keep_skey", &DataIO::SetKeepSKey)
    .def("pause", &DataIO::SetPause, 
         "pause reading after read limit sg, not exactly wait all sg exausted by default",
         pybind11::arg("limit"),
         pybind11::arg("wait_exactly")=false)
    .def("threads", &DataIO::SetThreads, "set threads", 
         pybind11::arg("threads"),
         pybind11::arg("threads_read")=8)
    .def("start_time", &DataIO::SetStartTime)
    .def("end_time", &DataIO::SetEndTime)
    .def("duration", &DataIO::SetDuration)
    .def("latest_time", &DataIO::GetLatestTime)
    .def("get_offset", &DataIO::GetReaderOffset)
    .def("sparse_list", &DataIO::sparse_list)
    .def("dense_list", &DataIO::dense_list)
    .def("ntable", &DataIO::ntable)
    .def("name", &DataIO::name)
    .def("fs", &DataIO::fs, pybind11::return_value_policy::reference_internal)
    .def("get_batch", &DataIO::GetBatch, pybind11::return_value_policy::reference)
    .def("serialize_state", [](DataIO *io){ return pybind11::bytes(io->Store()); })
    .def("restore_from_state", &DataIO::Restore)
    .def("feature", &DataIO::AddFeatureOpt, "add feature option",
         pybind11::arg("name"),
         pybind11::arg("type"),
         pybind11::arg("table")=0,
         pybind11::arg("nvec")=0,
         pybind11::arg("mask")="",
         pybind11::arg("serialized")=false,
         pybind11::arg("cutoff")=0,
         pybind11::arg("dsl")="");

  pybind11::class_<io::Operator>(m, "IOperator")
    .def("init", &Operator::Init)
    .def("urun", &Operator::URun);

  pybind11::class_<DebugOP, io::Operator>(m, "DebugOP")
    .def(pybind11::init<>())
    .def("init", &Operator::Init);

  pybind11::class_<IOPRegistry>(m, "IOPRegistry")
    .def(pybind11::init<>());

  m.def("GetIOP", &GetIOP, "Get registered data io ops");

  m.def("hdfs_write", &HdfsWrite, "write string to hdfs");
  m.def("hdfs_read", &HdfsRead, "read string from hdfs");
  m.def("get_file_system", &GetFileSystem, pybind11::return_value_policy::reference);
}

}  // namespace python_lib
}  // namespace xdl
