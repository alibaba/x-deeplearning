/*
 * \file predictor_example.cc 
 * \brief The predictor example
 */
#include "predictor.h"

#include <omp.h>

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "blaze/common/timer.h"

using namespace blaze;

#define DEBUG
//#define BENCH

#ifdef BENCH
size_t kBatchSize = 200;
#define kNumThreads 6
#else
size_t kBatchSize = 2;
#define kNumThreads 1
#endif

std::unordered_map<std::string, std::vector<uint64_t>> gIdsMap;
std::unordered_map<std::string, std::vector<float>> gValuesMap;
std::unordered_map<std::string, std::vector<uint32_t>> gSegmentsMap;

void InitMap() {
  uint64_t gids[] = { 3231764UL, 2144432UL, 815566UL, 1099441UL, 1971111UL, 1330483UL, 253802UL, 3817285UL, 1471227UL, 498101UL,
                      684658UL, 498101UL, 118517UL, 4890893UL, 1697189UL, 2329882UL, 132418UL, 2391981UL, 3392642UL, 2893478UL,
                      691792UL, 132418UL, 2641334UL, 4654145UL, 3358797UL, 4210985UL, 2017644UL, 3718203UL, 598920UL, 3392642UL,
                      378508UL, 3392642UL, 1847754UL, 4182683UL, 340969UL, 1847754UL, 1641934UL, 1797794UL, 598920UL, 3135125UL,
                      2813613UL, 340969UL, 1312031UL, 909025UL, 2641334UL, 598920UL, 1598127UL, 1190547UL, 5151948UL, 1494283UL,
                      5138337UL, 3051176UL, 2723807UL, 1136221UL, 1598127UL, 598920UL, 1743172UL, 5150705UL, 598920UL, 1140882UL,
                      4880802UL, 4778760UL, 2560123UL, 3392642UL, 3533123UL, 1950878UL, 3978525UL, 92831UL, 3533123UL, 
                    };

  for (int i = 1; i < 70; ++i) {
    std::string str_ids = std::string("item_") + std::to_string(i) + ".ids";
    std::string str_values = std::string("item_") + std::to_string(i) + ".values";
    std::string str_segments = std::string("item_") + std::to_string(i) + ".segments";
    
    gIdsMap[str_ids] = { gids[i - 1] };
    gValuesMap[str_values] = { 1.0f };
    gSegmentsMap[str_segments] = { 1 };
  }

  gIdsMap["unit_id_expand.ids"] = { 5163072UL, 5163073UL };
  gValuesMap["unit_id_expand.values"] = { 1.0f, 1.0f };
  gSegmentsMap["unit_id_expand.segments"] = { 1, 1 };

  for (auto k = 2; k < kBatchSize; ++k) {
    gIdsMap["unit_id_expand.ids"].push_back(5163073UL);
    gValuesMap["unit_id_expand.values"].push_back(1.0);
    gSegmentsMap["unit_id_expand.segments"].push_back(1);
  }
}

void FeedSparseFeature(const std::string& input_name,
                       const FeedNameConfig& feed_name_config,
                       Predictor* predictor) {
  switch (feed_name_config.sparse_feature_type) {
    case kSparseFeatureId:
      {
        auto ids = gIdsMap[input_name];
        predictor->ReshapeInput(input_name.c_str(), { ids.size() });
        predictor->Feed(input_name.c_str(), ids.data(), ids.size() * sizeof(int64_t));
      }
      break;
    case kSparseFeatureValue:
      {
        auto values = gValuesMap[input_name];
        predictor->ReshapeInput(input_name.c_str(), { values.size() });
        predictor->Feed(input_name.c_str(), values.data(), values.size() * sizeof(float));
      }
      break;
    case kAuxSparseFeatureSegment:
      {
        auto segments = gSegmentsMap[input_name];
        predictor->ReshapeInput(input_name.c_str(), { segments.size() });
        predictor->Feed(input_name.c_str(), segments.data(), segments.size() * sizeof(float));
      }
      break;
  }
}

void FeedAuxIndicator(const std::string& input_name,
                      const FeedNameConfig& feed_name_config,
                      Predictor* predictor) {
  std::vector<int32_t> indicator;
  indicator.resize(kBatchSize, 0);

  predictor->ReshapeInput(input_name.c_str(), { indicator.size() });
  predictor->Feed(input_name.c_str(), indicator.data(), indicator.size() * sizeof(int32_t));
}

static std::shared_ptr<PredictorManager> predictor_manager;

void DoPredictExample(const char* sparse_model_weight_uri, const char* model_file) {
  // Create predictor handle, You can create many predictor for
  // multi-thread envroment.
  auto predictor = predictor_manager->CreatePredictor(kPDT_CUDA);

#ifdef BENCH
  while (true) {
#endif
  // Feed data
  const auto& input_name_list = predictor->ListInputName();
  for (const auto& input_name : input_name_list) {
    auto feed_name_config = predictor->GetFeedNameConfig(input_name);
    switch (feed_name_config.feature_type) {
      case kDenseFeature:
        // In this example, we do not use dense feature
        break;
      case kSparseFeature:
        FeedSparseFeature(input_name, feed_name_config, predictor);
        break;
      case kAuxIndicator:
        FeedAuxIndicator(input_name, feed_name_config, predictor);
        break;
    }
  }

  // Forward
  Timer timer;
  timer.Start();
  for (int i = 0; i < 100; ++i) {
    predictor->Forward();
  }
  timer.Stop();

#ifdef BENCH
  }
#else
#ifdef DEBUG
  std::cout << "elapsed time:" << (timer.GetElapsedTime() / 100) << std::endl;
  // Step5: Fetch output
  float* data;
  size_t len = 1;
  auto success = predictor->Output("softmaxoutput0", reinterpret_cast<void**>(&data), &len);
  if (success) {
    size_t size = len / sizeof(float);
    for (auto i = 0; i < kBatchSize; ++i) {
      std::cout << "The " << i << "-th: ";
      for (auto j = 0; j < size / kBatchSize; ++j) {
        std::cout << data[i * size / kBatchSize + j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "predict success" << std::endl;
  } else {
    std::cerr << "predict failed" << std::endl;
  }

  // Step6: Print internal result
  auto internal_names = predictor->ListInternalName();
  std::cout << "internal_names.size()=" << internal_names.size() << std::endl;
  for (const auto& name : internal_names) {
    predictor->InternalParam(name.c_str(), reinterpret_cast<void**>(&data), &len);
    std::cout << "name:" << name << std::endl;
    size_t size = len / sizeof(float);
    for (auto i = 0; i < size; ++i) {
      std::cout << ((float*)data)[i] << " ";
    }
    std::cout << std::endl;
  }
#endif
#endif
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << argv[0] << " sparse_model_weight_uri model_file" << std::endl; 
    return 1;
  }
  InitMap();
  
  InitScheduler(false, 1000, 100, 32, 4, 2);

  predictor_manager.reset(new PredictorManager);
  predictor_manager->LoadSparseModelWeight(argv[1]);
  predictor_manager->LoadModel(argv[2]);
  predictor_manager->SetRunMode("hybrid");

  #pragma omp parallel for num_threads(kNumThreads)
  for (int i = 0; i < kNumThreads; ++i)
  {
    DoPredictExample(argv[1], argv[2]);
  }

  return 0;
}
