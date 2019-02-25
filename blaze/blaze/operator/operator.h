/*
 * \file operator.h
 * \desc The base operator.
 */
#pragma once

#include <string>
#include <map>

#include "blaze/common/context.h"
#include "blaze/common/event.h"
#include "blaze/common/exception.h"
#include "blaze/common/log.h"
#include "blaze/common/observer.h"
#include "blaze/common/proto_helper.h"
#include "blaze/common/registry.h"
#include "blaze/common/blob.h"
#include "blaze/common/types.h"
#include "blaze/graph/workspace.h"
#include "blaze/operator/operator_schema.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

class OperatorBase;
typedef ObserverBase<OperatorBase> OperatorObserver;

class OperatorBase : public Observable<OperatorBase> {
 public:
  explicit OperatorBase(const OperatorDef& def, Workspace* workspace, bool alloc_input_output_blob = true);
  virtual ~OperatorBase() noexcept { }

  // Checks if the operator has an argument of the given name.
  inline bool HasArgument(const std::string& name) const {
    return ArgumentHelper::HasArgument(def_, name);
  }
  // Functions that deal with arguments. Basically, this allows us to map
  // an argument name to a specific type of argument that we are typing to
  // access.
  template <typename T>
  inline T GetSingleArgument(const std::string& name, const T& default_value) const {
    return ArgumentHelper::GetSingleArgument<OperatorDef, T>(def_, name, default_value);
  }
  template <typename T>
  inline bool HasSingleArgumentOfType(const std::string& name) const {
    return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(def_, name);
  }

  template <typename T>
  inline std::vector<T> GetRepeatedArgument(const std::string& name,
                                            const std::vector<T>& default_value = {}) const {
    return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(def_, name, default_value);
  }

  // Run operator on device's stream_id
  virtual bool Run(int stream_id = 0) {
    LOG_ERROR("Not implemented");
    return false;
  }
  virtual void WaitStream(int stream_id = -1) { }

  // Return if the stream is free for task scheduling.
  // Used in stream allocation optimization to skip stream that is currently
  // busy.
  virtual bool StreamIsFree(int stream_id) {
    return true;
  }

  virtual void WaitEvent(const Event& ev, int stream_id = 0) {
    ev.Finish();
  }
  virtual void WaitEvents(const std::vector<const Event*>& events, int stream_id = 0) {
    for (const auto& ev : events) {
      ev->Finish();
    }
  }
  inline void Wait(const OperatorBase& other, int stream_id = 0) {
    if (!other.IsEventDisabled()) {
      WaitEvent(other.event(), stream_id);
    }
  }

  virtual void Finish() {
    if (event_) {
      event_->Finish();
    }
  }
  virtual void RecordEvent(const char* err_msg = nullptr) { }
  void DisableEvent() { event_ = nullptr; }
  bool IsEventDisabled() const { return !event_; }
  const Event& event() const { return *event_; }
  Event& event() { return *event_; }
  const DeviceOption& device_option() const { return device_option_; }
  // set the net position.
  void set_net_position(int net_position) { net_position_ = net_position; }
  int net_position() const { return net_position_; }

  virtual void ResetEvent() {
    if (event_) {
      event_->Reset();
    }
  }
  const OperatorDef& operator_def() const { return def_; }
  const std::string& type() const { return type_; }
  const std::string& name() const { return name_; }

  inline int InputSize() const { return inputs_.size(); }
  inline int OutputSize() const { return outputs_.size(); }

  inline std::vector<Blob*>& Inputs() { return inputs_; }
  inline std::vector<Blob*>& Outputs() { return outputs_; }

  // whether enable aync.
  void set_enable_async(bool enable_async) { enable_async_ = enable_async; }
  bool enable_async() const { return enable_async_; }

  inline Blob* Input(int idx) {
    CHECK(idx >= 0 && idx < inputs_.size(), "input idx=", idx, " size=", inputs_.size(), def_.DebugString());
    return inputs_[idx];
  }
  inline Blob* Output(int idx) {
    CHECK(idx >= 0 && idx < outputs_.size(), "output idx=", idx, " size=", outputs_.size(), def_.DebugString());
    return outputs_[idx];
  }
  virtual int stream_id() = 0;

 protected:
  // allocate blob for input and output
  void AllocInputOutputBlob(const OperatorDef& def);
  // check schema
  void CheckSchema(OpSchema* schema);

  OperatorDef def_;
  Workspace* workspace_;
  DeviceOption device_option_;
  std::string type_;
  std::string name_;

  // An event used by asynchronous execution
  std::unique_ptr<Event> event_; 

  // The inputs and outputs for the Operator
  std::vector<Blob*> inputs_;
  std::vector<Blob*> outputs_;

  // The net position
  int net_position_;
  bool enable_async_ = true;
};

template <class Context>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OperatorDef& def, Workspace* workspace, bool alloc_blob = true) :
      OperatorBase(def, workspace, alloc_blob), context_(def.device_option()) {
    context_.SwitchToDevice(0);
  }
  ~Operator() noexcept override {}

  void WaitEvent(const Event& ev, int stream_id = 0) final {
    if (stream_id >= 0) {
      context_.SwitchToDevice(stream_id);
    }
    context_.WaitEvent(ev);
  }
  void WaitEvents(const std::vector<const Event*>& events, int stream_id = 0) final {
    if (stream_id >= 0) {
      context_.SwitchToDevice(stream_id);
    }
    for (const auto& ev : events) {
      context_.WaitEvent(*ev);
    }
  }

  bool Run(int stream_id = 0) final {
    StartAllObservers();

    context_.SwitchToDevice(stream_id);
    bool result = RunOnDevice();
    if (context_.HasAsyncPart() && enable_async()) {
      RecordEvent();  // Record the event
    } else {
      SetEventFinished();
    }
    
    StopAllObservers();
    return result;
  }

  virtual bool RunOnDevice() = 0;

  bool StreamIsFree(int stream_id) final {
    return context_.StreamIsFree(device_option(), stream_id);
  }

  void RecordEvent(const char* err_msg = nullptr) final {
    if (event_) {
      context_.Record(event_.get(), err_msg);
    }
  }
  void SetEventFinished(const char* err_msg = nullptr) {
    if (event_) {
      event_->SetFinished(err_msg);
    }
  }
  // Return the context
  const Context& context() const { return context_; }
  // Return stream id
  virtual int stream_id() { return context_.stream_id(); }
  // Wait stream, if stream_id < 0, wait the current used stream id.
  virtual void WaitStream(int stream_id = -1) {
    if (stream_id < 0) stream_id = context_.stream_id();
    int old_stream_id = context_.stream_id();
    context_.SwitchToDevice(stream_id);
    context_.FinishDeviceComputation();
    context_.SwitchToDevice(old_stream_id);
  }

 protected:
  Context context_;
};

#define USE_OPERATOR_BASE_FUNCTIONS                            \
    using OperatorBase::HasArgument;                           \
    using OperatorBase::GetSingleArgument;                     \
    using OperatorBase::HasSingleArgumentOfType;               \
    using OperatorBase::GetRepeatedArgument;                   \
    using OperatorBase::InputSize;                             \
    using OperatorBase::OutputSize

#define USE_OPERATOR_FUNCTIONS(context)                        \
    USE_OPERATOR_BASE_FUNCTIONS;                               \
    using Operator<context>::context_;                         \
    using Operator<context>::Input;                            \
    using Operator<context>::Inputs;                           \
    using Operator<context>::Output;                           \
    using Operator<context>::Outputs

// The following are Registry utility of Operator
typedef Registry<std::unique_ptr<OperatorBase>,
                 const OperatorDef,
                 Workspace*> OperatorRegistry;
std::map<int, OperatorRegistry*>* GetDeviceTypeRegistry();

typedef Registry<std::unique_ptr<OperatorBase>,
                 const OperatorDef,
                 Workspace*>* (*RegistryCreateFunction)();

struct DeviceTypeRegistryCreator {
  explicit DeviceTypeRegistryCreator(int type, RegistryCreateFunction func) {
    if (GetDeviceTypeRegistry()->count(type)) {
      BLAZE_THROW("Device type ",
                  type,
                  " registered twice func = ",
                  func);
    }
    GetDeviceTypeRegistry()->emplace(type, func());
  }
};

#define REGISTER_DEVICE_TYPE(type, registry_create_function)        \
    namespace {                                                     \
      static DeviceTypeRegistryCreator ANONYMOUS_VARIABLE(          \
        DeviceType)(type, &registry_create_function);               \
    }

DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef,
    Workspace*);

#define REGISTER_CPU_OPERATOR(name, ...)                            \
    extern void PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();            \
    static void UNUSED ANONYMOUS_VARIABLE_CPU##name() {             \
      PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                      \
    }                                                               \
    REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

DECLARE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef,
    Workspace*);

#define REGISTER_CUDA_OPERATOR(name, ...)                           \
    extern void PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();            \
    static void UNUSED ANONYMOUS_VARIABLE_CUDA##name() {            \
      PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                      \
    }                                                               \
    REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

// Create an operator with the given operator definition
std::unique_ptr<OperatorBase> CreateOperator(const OperatorDef& def, Workspace* ws, int new_position);

}  // namespace blaze
