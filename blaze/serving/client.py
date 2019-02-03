#!/usr/bin/python
# -*- coding: UTF-8 -*-

import urllib2
import time
# import predict_pb2

json_req = '''
{
  "model_version": "tdm_dnn",
  "feature_name" : [
    "item_1",
    "unit_id_expand"
  ],
  "ad_feature" : [
    {
      "tensor": [ 
        {
          "feature_name_index": 1,
          "key": [5163072],
          "value": [1.0],
        }
      ]
    }
  ],
  "user_feature" : {
    "tensor" : [
      {
        "feature_name_index": 0,
        "key": [3231764],
        "value": [1.0],
      }
    ]
  }
}
'''


def parseText():
    requrl = "http://127.0.0.1:8080/predict"
    req = urllib2.Request(requrl, json_req)
    res_data = urllib2.urlopen(req)
    res = res_data.read()
    print res

def main():
        time1 = time.time()
        parseText()
        time2 = time.time()
        print "rt: "+str((time2 - time1)*1000)+"ms"

if __name__ == '__main__':
        main()

