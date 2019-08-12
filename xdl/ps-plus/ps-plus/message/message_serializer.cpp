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

#include "message_serializer.h"

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::ServerInfo>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::ServerInfo>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::ClusterInfo>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::ClusterInfo>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::VariableInfo>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::VariableInfo>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::VariableInfoCollection>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::VariableInfoCollection>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::UdfChainRegister::UdfDef>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::UdfChainRegister::UdfDef>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::UdfChainRegister>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::UdfChainRegister>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::DenseVarNames>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::DenseVarNames>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::DenseVarValues>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::DenseVarValues>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::WorkerState>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::WorkerState>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<ps::WorkerState> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<ps::WorkerState> >);

