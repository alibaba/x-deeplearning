from pyblaze.predictor import PredictorManager
from pyblaze.predictor import Predictor
import numpy as np
import unittest

qed_file = './sparse_qed'
model_file = './model_blaze_optimized'

batch_size = 4
indicator_shape = [ batch_size ]
indicator = np.zeros([batch_size], np.int32)

## ncomm data
ncomm_id_shape = [ 2 * batch_size ]
ncomm_value_shape = [ 2 * batch_size ]
ncomm_segments_shape = [ batch_size ]

ncomm_id = np.ones(ncomm_id_shape, np.int64)
ncomm_value = np.ones(ncomm_value_shape, np.float)
ncomm_segment = 2 * np.ones(ncomm_segments_shape, np.int32)

## comm data
comm_id_shape = [ 2 ]
comm_value_shape = [ 2 ]
comm_segments_shape = [ 1 ]

comm_id = np.ones(comm_id_shape, np.int64)
comm_value = np.ones(comm_value_shape, np.float)
comm_segment = 2 * np.ones(comm_segments_shape, np.int32)

def predict_model(optimization_pass, device_type, device_id):
    pm = PredictorManager()
    pm.load_sparse_model_weight(qed_file)
    pm.load_model(model_file, optimization_pass)

    predictor = pm.create_predictor(device_type, device_id)

    print 'input len=', len(predictor.list_input_names())
    for feed_name in predictor.list_input_names():
      splits = feed_name.split('.')
      print splits[0]
    
    ## feed indicator
    predictor.reshape_input("indicator.0", indicator_shape)
    predictor.feed("indicator.0", indicator)

    ## feed comm sparse feature
    for x in xrange(1, 70):
      name_ids = 'item_%d.ids' % (x)
      predictor.reshape_input(name_ids, comm_id_shape)
      predictor.feed(name_ids, comm_id)

      name_values = 'item_%d.values' % (x)
      predictor.reshape_input(name_values, comm_value_shape)
      predictor.feed(name_values, comm_value)

      name_segments = 'item_%d.segments' % (x)
      predictor.reshape_input(name_segments, comm_segments_shape)
      predictor.feed(name_segments, comm_segment)

    # feed ncomm sparse feature
    name_ids = 'unit_id_expand.ids'
    predictor.reshape_input(name_ids, ncomm_id_shape)
    predictor.feed(name_ids, ncomm_id)

    name_values = 'unit_id_expand.values'
    predictor.reshape_input(name_values, ncomm_value_shape)
    predictor.feed(name_values, ncomm_value)

    name_segments = 'unit_id_expand.segments'
    predictor.reshape_input(name_segments, ncomm_segments_shape)
    predictor.feed(name_segments, ncomm_segment)

    predictor.forward()

    print predictor.output_asnumpy('softmaxoutput0')

    names = predictor.list_internal_names()
    for name in names:
      data = predictor.internal_asnumpy(name)
      print 'name=', name, data

predict_model(1, 0, 0)
