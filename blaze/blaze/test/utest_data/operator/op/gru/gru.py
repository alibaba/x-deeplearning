import numpy as np
import blaze_pb2 as b

def sigmoid(a): return 1.0 / (1.0 + np.exp(-a))

def calc_gru_1(inp, h, i2h, i2h_bias, h2h, h2h_bias):
    preact = np.matmul(inp, i2h)
    if np.count_nonzero(preact) == 0: return np.zeros(inp.shape())
    h_p = np.matmul(h, h2h) + h2h_bias
    br, bz, bh = np.split(i2h_bias, 3)
    r0, z0, h0 = np.split(preact, 3)
    r1, z1, h1 = np.split(h_p, 3)
    r2 = np.apply_along_axis(sigmoid, 0, r0 + r1 + br)
    z2 = np.apply_along_axis(sigmoid, 0, z0 + z1 + bz)
    h2 = np.tanh(h0 + h1 * r2 + bh)
    return (np.ones(inp.shape) + np.negative(z2)) * h2 + z2 * h

def is_fp16(dtype): return dtype == 12

def random_numbers(shape, dtype):
    scale = 1e-2
    if isinstance(shape, int):
        vs = (np.random.random_sample(shape) * 2 - 1) * scale
    else:
        n = reduce(lambda x, y: x * y, shape)
        vs = (np.random.random_sample(n) * 2 - 1).reshape(shape) * scale
    return np.float16(vs) if is_fp16(dtype) else vs

def calc_gru(dtype, batch_size, seq_len, num_hidden):
    i2h = random_numbers((num_hidden, num_hidden * 3), dtype)
    i2h_bias = random_numbers((num_hidden * 3), dtype)
    h2h = random_numbers((num_hidden, num_hidden * 3), dtype)
    h2h_bias = random_numbers((num_hidden * 3), dtype)
    input = random_numbers((batch_size, seq_len, num_hidden), dtype)
    result = np.array([]).reshape((0, seq_len, num_hidden))
    for inp in np.split(input, batch_size):
        inp = np.squeeze(inp, 0)
        out = np.array([]).reshape((0, num_hidden))
        o = np.zeros((num_hidden))
        for i0 in np.split(inp, seq_len):
            i = np.squeeze(i0, 0)
            o = calc_gru_1(i, o, i2h, i2h_bias, h2h, h2h_bias)
            out = np.concatenate((out, np.expand_dims(o, 0)), 0)
        result = np.concatenate((result, np.expand_dims(out, 0)), 0)
    return input, i2h, i2h_bias, h2h, h2h_bias, result

def tolist(vs): return vs.flatten().tolist()

def gen_input_op(name, shape, type, vs):
    op = b.OperatorDef()
    op.name = name
    op.type = "ConstantFill"
    a = op.arg.add()
    a.name = "shape"
    a.ints.extend(shape)
    a = op.arg.add()
    a.name = "dtype"
    a.i = type
    a = op.arg.add()
    a.name = "value"
    a.floats.extend(vs)
    op.output.extend([name])
    return op

def gen_gru(name, dtype, batch_size, seq_len, num_hidden, from_deepnet = True):
    input, i2h, i2h_bias, h2h, h2h_bias, result = calc_gru(
        dtype, batch_size, seq_len, num_hidden)
    if from_deepnet:
        input_ = gen_input_op(
            "input", [batch_size, seq_len, num_hidden], dtype, tolist(input))
        i2h_ = gen_input_op(
            "i2h", [num_hidden, num_hidden * 3], dtype, tolist(i2h))
        i2h_bias_ = gen_input_op(
            "i2h_bias", [num_hidden * 3], dtype, tolist(i2h_bias))
        h2h_ = gen_input_op(
            "h2h", [num_hidden, num_hidden * 3], dtype, tolist(h2h))
        h2h_bias_ = gen_input_op(
            "h2h_bias", [num_hidden * 3], dtype, tolist(h2h_bias))
        input_ops = [input_, h2h_, i2h_, h2h_bias_, i2h_bias_]
        op = b.OperatorDef()
        op.name = "GRU"
        op.type = "GRU"
        op.input.extend([xxx.name for xxx in input_ops])
        op.output.extend(["output"])
        res = op.arg.add()
        res.name = "from_deepnet"
        res.i = 1
        net = b.NetDef()
        net.name = name
        net.run_mode = "simple"
        net.op.extend(input_ops + [op])
        res = net.external_output.add()
        res.name = "output"
        res.dtype = dtype
        res = net.arg.add()
        res.name = "output_shape"
        res.ints.extend([batch_size, seq_len, num_hidden])
        res = net.arg.add()
        res.name = "output"
        res.floats.extend(tolist(result))
        net.device_option.device_id = 0
        net.device_option.device_type = 0
        with open(name + '_cpu.conf', 'w') as f: f.write(str(net))
        net.device_option.device_type = 1
        with open(name + '_gpu.conf', 'w') as f: f.write(str(net))
    else:
        input_ = gen_input_op(
            "input", [batch_size, seq_len, num_hidden], dtype, tolist(input))
        i2h_ = gen_input_op(
            "i2h", [num_hidden, num_hidden * 3], dtype, tolist(i2h))
        h2h_ = gen_input_op(
            "h2h", [num_hidden, num_hidden * 3], dtype, tolist(h2h))
        bias_ = gen_input_op(
            "bias", [1, num_hidden * 6], dtype, tolist(
                np.concatenate((i2h_bias, h2h_bias), 0)))
        input_ops = [input_, i2h_, h2h_, bias_]
        op = b.OperatorDef()
        op.name = "GRU"
        op.type = "GRU"
        op.input.extend([xxx.name for xxx in input_ops])
        op.output.extend(["unused_output", "output"])
        net = b.NetDef()
        net.name = name
        net.run_mode = "simple"
        net.op.extend(input_ops + [op])
        res = net.external_output.add()
        res.name = "unused_output"
        res = net.external_output.add()
        res.name = "output"
        res.dtype = dtype
        res = net.arg.add()
        res.name = "output_shape"
        res.ints.extend([batch_size, seq_len, num_hidden])
        res = net.arg.add()
        res.name = "output"
        res.floats.extend(tolist(result))
        net.device_option.device_id = 0
        net.device_option.device_type = 0
        with open(name + '_cpu_onnx.conf', 'w') as f: f.write(str(net))
        net.device_option.device_type = 1
        with open(name + '_gpu_onnx.conf', 'w') as f: f.write(str(net))

np.random.seed(0)
gen_gru("gru_test_1_1_1", 1, 1, 1, 1)
gen_gru("gru_test_1_1_2", 1, 1, 1, 2)
gen_gru("gru_test_1_2_1", 1, 1, 2, 1)
gen_gru("gru_test_1_2_2", 1, 1, 2, 2)
gen_gru("gru_test_2_4_5", 1, 2, 4, 5)
gen_gru("gru_test_4_20_9", 1, 4, 20, 9)
gen_gru("gru_test_7_15_23", 1, 7, 15, 23)
gen_gru("gru_test_1_1_108", 1, 1, 1, 108)
gen_gru("gru_test_1_2_108", 1, 1, 2, 108)
gen_gru("gru_test_1_3_108", 1, 1, 3, 108)
gen_gru("gru_test_1_50_108", 1, 1, 50, 108)
gen_gru("gru_test_2_50_108", 1, 2, 50, 108)
gen_gru("gru_test_2_30_128", 1, 2, 30, 128)

gen_gru("gru_test_1_1_1", 1, 1, 1, 1, False)
gen_gru("gru_test_1_1_2", 1, 1, 1, 2, False)
gen_gru("gru_test_1_2_1", 1, 1, 2, 1, False)
gen_gru("gru_test_1_2_2", 1, 1, 2, 2, False)
gen_gru("gru_test_2_4_5", 1, 2, 4, 5, False)
gen_gru("gru_test_4_20_9", 1, 4, 20, 9, False)
gen_gru("gru_test_7_15_23", 1, 7, 15, 23, False)
gen_gru("gru_test_1_1_108", 1, 1, 1, 108, False)
gen_gru("gru_test_1_2_108", 1, 1, 2, 108, False)
gen_gru("gru_test_1_3_108", 1, 1, 3, 108, False)
gen_gru("gru_test_1_50_108", 1, 1, 50, 108, False)
gen_gru("gru_test_2_50_108", 1, 2, 50, 108, False)
gen_gru("gru_test_2_30_128", 1, 2, 30, 128, False)

gen_gru("gru_test_1_1_2_fp16", 12, 1, 1, 2)
gen_gru("gru_test_1_2_2_fp16", 12, 1, 2, 2)
