from pyblaze.predictor import PredictorManger
from pyblaze.predictor import Predictor
import numpy as np

batch_size = 1
comm_shape = [1, 306]
ncomm_shape = [batch_size, 732]
att_comm_shape = [1, 1800 * 18]

np.random.seed(0)
comm = 0.01 * np.random.random_sample(comm_shape).astype(np.float32)
ncomm = 0.01 * np.random.random_sample(ncomm_shape).astype(np.float32)
att_comm = 0.01 * np.random.random_sample(att_comm_shape).astype(np.float32)

def read_tensor(filename, np_array):
    with open(filename) as f:
        content = f.readline()
        values = content.split(',')
        for k in xrange(len(values) - 1):
            np_array[0][k] = float(values[k])
        print 'size=', len(values)

read_tensor('./comm.txt', comm)
read_tensor('./ncomm.txt', ncomm)
read_tensor('./att_comm.txt', att_comm)

conf_file = './net-parameter-conf'
param_file = './dnn-model-dat'

def predict_model(optimization_pass, device_type, device_id):
    pm = PredictorManger()
    pm.load_deepnet_model(conf_file, param_file, optimization_pass)

    predictor = pm.create_predictor(device_type, device_id)

    print predictor.list_input_names()
    print predictor.list_internal_names()
    predictor.register_observers(['cost'])

    predictor.reshape_input('comm', comm_shape)
    predictor.reshape_input('ncomm', ncomm_shape)
    predictor.reshape_input('att_comm', att_comm_shape)
    
    predictor.feed('comm', comm)
    predictor.feed('ncomm', ncomm)
    predictor.feed('att_comm', att_comm)

    predictor.forward()
    print predictor.output_asnumpy('output')
    predictor.forward()
    print predictor.output_asnumpy('output')
    return predictor.output_asnumpy('output')

### CPU No-OptPass
res0 = predict_model(0, 0, 0)
#print res0
### GPU No-OptPass
res1 = predict_model(1, 1, 0)
#print res1
res1 = predict_model(1, 1, 0)
print res1
