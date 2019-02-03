#ifndef __SERVING_MODEL_MANAGER_H__
#define __SERVING_MODEL_MANAGER_H__

#include <map>
#include <string>

#include "./model.h"


namespace serving {

class ModelManager {
public:
  static ModelManager* Instance();
  bool Init(const std::string& config_file);

//  bool Release();

  Predictor* CreatePredictor(const std::string& model_version);

private:
  ModelManager() {};
  virtual ~ModelManager() {};

  //disable copy and assignment
  ModelManager(const ModelManager &);
  ModelManager &operator=(const ModelManager &);

  std::map<std::string, Model> model_version_map_;
};

}//namespace serving

#endif
