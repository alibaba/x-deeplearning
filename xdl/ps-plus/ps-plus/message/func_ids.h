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

#ifndef PS_MESSAGE_FUNC_IDS_H_
#define PS_MESSAGE_FUNC_IDS_H_

namespace ps {
namespace func_ids {

static const int kSchedulerGetVersion               = 0x00010001;
static const int kSchedulerSave                     = 0x00010002;
static const int kSchedulerRestore                  = 0x00010003;
static const int kSchedulerRegisterServer           = 0x00010004;
static const int kSchedulerGetClusterInfo           = 0x00010005;
static const int kSchedulerUpdateVariableInfo       = 0x00010006;
static const int kSchedulerTriggerStreamingDense    = 0x00010007;
static const int kSchedulerTriggerStreamingSparse   = 0x00010008;
static const int kSchedulerTriggerStreamingHash     = 0x00010009;
static const int kSchedulerAsynchronizeEnter        = 0x0001000a;
static const int kSchedulerWorkerReportFinish       = 0x0001000b;
static const int kSchedulerSynchronizeEnter         = 0x0001000c;
static const int kSchedulerSynchronizeLeave         = 0x0001000d;
static const int kSchedulerUpdateVariableVisitInfo  = 0x0001000e;
static const int kSchedulerGetWorkerFinishCount     = 0x0001000f;
static const int kSchedulerWorkerBarrier            = 0x00010010;
static const int kSchedulerInitGlobalFileQueue      = 0x00010011;
static const int kSchedulerGetNextFile              = 0x00010012;
static const int kSchedulerReportWorkerState        = 0x00010013;
static const int kSchedulerRestoreWorkerState       = 0x00010014;
static const int kSchedulerWorkerBarrierV2          = 0x00010015;

static const int kServerRegisterUdfChain            = 0x00020001;
static const int kServerProcess                     = 0x00020002;
static const int kServerSave                        = 0x00020003;
static const int kServerRestore                     = 0x00020004;
static const int kServerAnnounce                    = 0x00020005;
static const int kServerStreamingDenseVarName       = 0x00020006;
static const int kServerGatherStreamingDenseVar     = 0x00020007;
static const int kServerTriggerStreamingSparse      = 0x00020008;
static const int kServerTriggerStreamingHash        = 0x00020009;

static const int kModelServerFlush                  = 0x00030001;
static const int kModelServerForward                = 0x00030002;
static const int kModelServerBackward               = 0x00030003;

}
}

#endif // PS_MESSAGE_FUNC_IDS_H_
