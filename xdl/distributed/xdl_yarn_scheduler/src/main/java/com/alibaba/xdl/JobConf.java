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

package com.alibaba.xdl;

import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("unchecked")
public class JobConf {

  private Map<String, Object> jobConf;

  public JobConf(Map<String, Object> jobConf) {
    this.jobConf = jobConf;
  }

  public Map<String, Object> getJobConf() {
    return jobConf;
  }

  public String getSummaryOutputDir() {
    return this.getStringValue("summary", "output_dir");
  }

  public void setSummaryOutputDir(String value) {
    this.setValue("summary", "output_dir", value);
  }

  public String getCheckpointOutputDir() {
    return this.getStringValue("checkpoint", "output_dir");
  }

  public void setCheckpointOutputDir(String value) {
    this.setValue("checkpoint", "output_dir", value);
  }

  public String getAucScoreOutputDir() {
    return this.getStringValue("auc", "auc_score_output_dir");
  }

  public void setAucScoreOutputDir(String dir) {
    this.setValue("auc", "auc_score_output_dir", dir);
  }

  public Object getValue(String key) {
    return jobConf.get(key);
  }

  public Object getValue(String first, String second) {
    Map<String, Object> firstVal = (Map<String, Object>) jobConf.get(first);
    if (firstVal != null && firstVal.containsKey(second)) {
      return firstVal.get(second);
    }
    return null;
  }

  public void setValue(String first, String second, Object value) {
    Map<String, Object> firstVal = (Map<String, Object>) jobConf.get(first);
    if (firstVal == null) {
      firstVal = new HashMap<String, Object>();
      jobConf.put(first, firstVal);
    }
    firstVal.put(second, value);
  }

  public String getStringValue(String key) {
    Object val = this.getValue(key);
    if (val == null) {
      return null;
    } else {
      return val.toString();
    }
  }

  public String getStringValueDef(String key, String def) {
    Object val = this.getValue(key);
    if (val == null) {
      return def;
    } else {
      return val.toString();
    }
  }

  public String getStringValue(String first, String second) {
    Object val = this.getValue(first, second);
    if (val == null) {
      return null;
    } else {
      return val.toString();
    }
  }

  public String getStringValueDef(String first, String second, String def) {
    Object val = this.getValue(first, second);
    if (val == null) {
      return def;
    } else {
      return val.toString();
    }
  }
}
