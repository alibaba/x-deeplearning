from pyblaze.predictor import PredictorManger
from pyblaze.predictor import Predictor
import numpy as np
import time

batch_size = 200
comm_shape = [1, 306]
ncomm_shape = [batch_size, 732]
att_comm_shape = [1, 1800 * 18]

np.random.seed(0)
comm = 0.01 * np.random.random_sample(comm_shape).astype(np.float32)
ncomm = 0.01 * np.random.random_sample(ncomm_shape).astype(np.float32)
att_comm = 0.01 * np.random.random_sample(att_comm_shape).astype(np.float32)

conf_file = './net-parameter-conf'
param_file = './dnn-model-dat'

def predict_model(optimization_pass, device_type, device_id):
    pm = PredictorManger()
    pm.load_deepnet_model(conf_file, param_file, optimization_pass)

    predictor = pm.create_predictor(device_type, device_id)

    print predictor.list_input_names()
    #predictor.register_observers(['cost'])

    predictor.reshape_input('comm', comm_shape)
    predictor.reshape_input('ncomm', ncomm_shape)
    predictor.reshape_input('att_comm', att_comm_shape)
    
    predictor.feed('comm', comm)
    predictor.feed('ncomm', ncomm)
    predictor.feed('att_comm', att_comm)

    count = 0
    start_time = time.time()
    while count <= 1000 * 1000:
        predictor.forward()
        count += 1
        if count % 1000 == 0:
            print 'qps=', count / (time.time() - start_time)
            #print predictor.dump_observers()

while True:
    res3 = predict_model(1, 1, 1)

