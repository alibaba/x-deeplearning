/*
 * \file embedding_op.h 
 * \brief The embedding operation, pull sparse parameter from local storage or
 * distributed storage.
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"
#include "blaze/proto/embedding.pb.h"
#include "blaze/store/sparse_puller.h"
#include "blaze/operator/common_helper.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
struct SparseInputItem {
  K_DType* ids;
  V_DType* values;
  N_DType* id_nums;
  size_t num_size;
};

template <typename V_DType>
struct EmbeddingBlockOutput {
  V_DType* values;
};

template <typename K_DType, typename V_DType, typename N_DType>
struct EmbeddingParam {
  std::vector<SparseInputItem<K_DType, V_DType, N_DType> > input_items;
  std::vector<EmbeddingBlockOutput<V_DType> > output_blocks;
};

template <typename Context>
class EmbeddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  EmbeddingOp(const OperatorDef& def, Workspace* workspace);
  bool RunOnDevice() override;

 protected:
  // init by embedding config
  void Init(const EmbeddingConfig& embedding_config) {
    embedding_config_.reset(new EmbeddingConfig(embedding_config));
    InitFeatureGroupConfigMeta();
    InitEmbeddingBlockConfigMeta();
  }

  // init embedding feature group config
  void InitFeatureGroupConfigMeta() {
    auto feature_group_size = embedding_config_->feature_group_config_size();
    efgcm_.resize(feature_group_size);
    for (auto i = 0; i < feature_group_size; ++i) {
      EmbeddingFeatureGroupConfigMeta &fg_config_meta = efgcm_[i];
      fg_config_meta.efgc = embedding_config_->feature_group_config(i);
      const std::string &feature_group = fg_config_meta.efgc.feature_group();
      fg_config_idx_map_[feature_group] = i;
    }
  }

  // init embedding block config
  void InitEmbeddingBlockConfigMeta() {
    // calc embeedding block offset & stride
    auto block_size = embedding_config_->block_config_size();
    for (auto i = 0; i < block_size; ++i) {
      auto embed_block_config = embedding_config_->mutable_block_config(i);
      size_t offset = 0;
      for (auto j = 0; j < embed_block_config->embedding_block_config_item_size(); ++j) {
        auto block_config_item = embed_block_config->mutable_embedding_block_config_item(j);
        const std::string& feature_group = block_config_item->feature_group();

        auto iter = fg_config_idx_map_.find(feature_group);
        BLAZE_CONDITION_THROW(iter != fg_config_idx_map_.end(),
                              "feature group config not exist, fg: ",
                              feature_group);
        auto& fg_config_meta = efgcm_[iter->second];
        fg_config_meta.block_infos.emplace_back(i, j);
        block_config_item->set_offset(offset);
        offset += GetEmbeddingBlockItemStride(*block_config_item,
                                              fg_config_meta.efgc.dim());
      }
      embed_block_config->set_stride(offset);
    }

    // init embedding block config meta
    ebcm_.resize(block_size);
    for (auto i = 0; i < block_size; ++i) {
      EmbeddingBlockConfigMeta& block_config_meta = ebcm_[i];
      block_config_meta.ebc = embedding_config_->block_config(i);
    }
  }

  // get block item stride, return stride element num
  int GetEmbeddingBlockItemStride(const EmbeddingBlockConfigItem& block_config_item,
                                  int dim) {
    switch (block_config_item.udf_type()) {
      case UDFType::kSum:
      case UDFType::kAvg:
        return dim;
      case UDFType::kAssign:
        return dim * block_config_item.trunc_num();
      default:
        BLAZE_THROW("Unsupported udf type: ", block_config_item.udf_type());
    }
  }

  void CheckValid() {
    // check input size
    BLAZE_CONDITION_THROW(this->InputSize() % 3 == 0,
                          "embedding op input size=",
                          this->InputSize());

    // check data type
    for (auto i = 0; i < this->InputSize(); ++i) {
      Blob* x = this->Input(i % 3);
      Blob* y = this->Input(i);
      BLAZE_CONDITION_THROW(x->data_type() == y->data_type(),
                            "x->data_type()=",
                            x->data_type(),
                            " y->data_type()=",
                            y->data_type());
    }

    // id & value size must be equal
    for (auto i = 0; i < this->InputSize() / 3; ++i) {
      Blob* id_blob = this->Input(i * 3);
      Blob* value_blob = this->Input(i * 3 + 1);
      BLAZE_CONDITION_THROW(id_blob->size() == value_blob->size(),
                            "id_blob->size()=",
                            id_blob->size(),
                            " value_blob->size()=",
                            value_blob->size(), this->def_.DebugString());
    }
  }

  // Prepare embedding param
  template <typename K_DType, typename V_DType, typename N_DType>
  void Setup(EmbeddingParam<K_DType, V_DType, N_DType>* params) {
    // prepare input items
    params->input_items.resize(this->InputSize() / 3);
    for (auto i = 0, p = 0; i < this->InputSize(); i += 3, ++p) {
      Blob *id_blob = this->Input(i);
      params->input_items[p].ids = id_blob->as<K_DType>();
      Blob *value_blob = this->Input(i + 1);
      params->input_items[p].values = value_blob->as<V_DType>();
      Blob *num_blob = this->Input(i + 2);
      params->input_items[p].id_nums = num_blob->as<N_DType>();
      params->input_items[p].num_size = num_blob->size();
    }

    // prepare output blocks
    params->output_blocks.resize(ebcm_.size());
    for (auto i = 0; i < ebcm_.size(); ++i) {
      size_t batch_size = 0;
      size_t stride = 0;
      const auto &embed_block_config = ebcm_[i].ebc;
      for (const auto &block_config_item : embed_block_config.embedding_block_config_item()) {
        const std::string &feature_group = block_config_item.feature_group();
        auto iter = input_fg_idx_map_.find(feature_group);
        BLAZE_CONDITION_THROW(iter != input_fg_idx_map_.end(),
                              "feature group of block item is not in input list, fg: ",
                              feature_group);
        auto idx = iter->second;
        if (batch_size != 0 && params->input_items[idx].num_size != batch_size) {
          BLAZE_THROW("batch size of each feature group in block must be equal");
        }

        if (params->input_items[idx].num_size != batch_size) {
          batch_size = params->input_items[idx].num_size;
        }
      }

      Blob *out_blob = this->Output(i);
      std::vector<TIndex> shape(2);
      shape[0] = static_cast<TIndex>(batch_size);
      shape[1] = static_cast<TIndex>(embed_block_config.stride());
      out_blob->Reshape(shape);
      memset(out_blob->as<char*>(), 0, out_blob->size() * DataTypeSize(out_blob->data_type()));
      params->output_blocks[i].values = out_blob->as<V_DType>();
    }
  }

  // Run embedding impl
  template <typename K_DType, typename V_DType, typename N_DType>
  void RunEmbedding(EmbeddingParam<K_DType, V_DType, N_DType>& params,
                    int key_type,
                    int value_type,
                    int num_type) {
    // Step 1: prepare sparse puller input & output
    auto feature_group_size = efgcm_.size();
    std::vector<store::SparsePullerInput> sparse_puller_inputs(feature_group_size);
    std::vector<store::SparsePullerOutput> sparse_puller_outputs(feature_group_size);
    for (auto i = 0; i < feature_group_size; ++i) {
      const auto& fg_config_meta = efgcm_[i];
      store::SparsePullerInput& sparse_puller_input = sparse_puller_inputs[i];
      store::SparsePullerOutput& sparse_puller_output = sparse_puller_outputs[i];
      // set common params
      const std::string& table_name = fg_config_meta.efgc.table_name();
      sparse_puller_input.name = table_name;
      const std::string& feature_group = fg_config_meta.efgc.feature_group();
      auto iter = input_fg_idx_map_.find(feature_group);
      if (iter == input_fg_idx_map_.end()) {
        continue;
      }

      auto idx = iter->second;
      sparse_puller_input.key = params.input_items[idx].ids;
      sparse_puller_input.key_num = params.input_items[idx].id_nums;
      sparse_puller_input.key_num_size = params.input_items[idx].num_size;
      sparse_puller_input.value = params.input_items[idx].values;
      sparse_puller_input.key_type = key_type;
      sparse_puller_input.value_type = value_type;
      sparse_puller_input.num_type = num_type;
      sparse_puller_input.dim = fg_config_meta.efgc.dim();

      // set block params
      sparse_puller_input.in_item.resize(fg_config_meta.block_infos.size());
      sparse_puller_output.out_item.resize(fg_config_meta.block_infos.size());
      for (auto j = 0; j < fg_config_meta.block_infos.size(); ++j) {
        auto block_idx = fg_config_meta.block_infos[j].block_idx;
        auto block_item_config_idx = fg_config_meta.block_infos[j].item_config_idx;
        const auto& embed_block_config_item =
            ebcm_[block_idx].ebc.embedding_block_config_item(block_item_config_idx);

        // set in_item param
        store::SparsePullerInput::Param& param = sparse_puller_input.in_item[j];
        param.udf_type = static_cast<store::UDFType>(embed_block_config_item.udf_type());
        param.trunc_direction =
            static_cast<store::TruncDirection>(embed_block_config_item.trunc_direction());
        param.trunc_num = embed_block_config_item.trunc_num();

        // set out_item out offset & stride
        store::SparsePullerOutput::OutItem& out_item = sparse_puller_output.out_item[j];
        auto* offset = params.output_blocks[block_idx].values + embed_block_config_item.offset();
        out_item.out = reinterpret_cast<void*>(offset);
        out_item.stride = ebcm_[block_idx].ebc.stride();
      }
    }

    auto status = sparse_puller_->Get(sparse_puller_inputs, sparse_puller_outputs);
    BLAZE_CONDITION_THROW(status == store::kOK,
                          "pull data failed, status=",
                          status);
  }

  // sparse puller
  std::shared_ptr<SparsePuller> sparse_puller_;

  // fg config
  struct EmbeddingFeatureGroupConfigMeta {
    EmbeddingFeatureGroupConfig efgc;
    struct BlockInfo {
      BlockInfo(int i, int j) :
          block_idx(i), item_config_idx(j) {}
      int block_idx;
      int item_config_idx;
    };
    std::vector<BlockInfo> block_infos;
  };
  std::vector<EmbeddingFeatureGroupConfigMeta> efgcm_;

  // block config
  struct EmbeddingBlockConfigMeta {
    EmbeddingBlockConfig ebc;
  };
  std::vector<EmbeddingBlockConfigMeta> ebcm_;

  // feature_group -> op input idx
  std::unordered_map<std::string, size_t> input_fg_idx_map_;
  // feature_group -> efgcm idx
  std::unordered_map<std::string, size_t> fg_config_idx_map_;
  // embedding config parsed from pb config file
  std::shared_ptr<EmbeddingConfig> embedding_config_;
};

}  // namespace blaze
