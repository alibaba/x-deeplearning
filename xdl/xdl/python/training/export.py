import multiprocessing
import numpy as np
import os
import struct
import sys
import xdl

def _popen(cmd):
    print 'os.popen(%s)' % cmd
    return os.popen(cmd)

def _system(cmd):
    print 'os.system(%s)' % cmd
    return os.system(cmd)

def _cmd(cmd):
    ret = _system(cmd)
    if ret != 0:
        raise ValueError("Failed: %s" % cmd)

def _is_hdfs(dir):
    return dir.startswith('hdfs:')

def _put_to_hdfs(output_dir, file):
    def _mkdir_hdfs(output_dir):
        ret = _system('hadoop fs -ls %s 1>/dev/null' % output_dir)
        if ret != 0:
            _system('hadoop fs -mkdir %s' % output_dir)
    _mkdir_hdfs(output_dir)
    _cmd('hadoop fs -put -f %s %s' % (file, output_dir))

def _mv_to_dir(output_dir, file):
    def _mkdir(output_dir):
        ret = _system('ls %s 1>/dev/null' % output_dir)
        if ret != 0:
            _cmd('mkdir -p %s' % output_dir)
    _mkdir(output_dir)
    _cmd('mv %s %s' % (file, output_dir))

def output(output_dir, fname):
    md5 = fname + '.md5'
    _cmd("md5sum %s |awk '{print $1}' >%s" % (fname, md5))
    if _is_hdfs(output_dir):
        _put_to_hdfs(output_dir, fname)
        _put_to_hdfs(output_dir, md5)
    else:
        _mv_to_dir(output_dir, fname)
        _mv_to_dir(output_dir, md5)

def get_latest_ckpt_v2(ckpt_dir):
    def _readSize(f):
        return struct.unpack("Q", f.read(8))[0]
    def _readStr(f):
        str_len = _readSize(f)
        return struct.unpack(str(str_len) + "s", f.read(str_len))[0]
    
    ret = 0
    checkpoints = os.path.join(ckpt_dir, 'checkpoints')
    if _is_hdfs(ckpt_dir):
        _system('rm -f checkpoints')
        ret = _system('hadoop fs -get %s 2>/dev/null' % checkpoints)
        checkpoints = 'checkpoints'
    else:
        ret = _system('ls %s 2>/dev/null' % checkpoints)
    if ret != 0:
        return ckpt_dir
    
    with open(checkpoints, "rb") as f:
        size = _readSize(f)
        last_ckpt = ""
        for i in range(size):
            last_ckpt = _readStr(f)
        return os.path.join(ckpt_dir, last_ckpt)

def _read_dense_file(f):
    vartype = struct.Struct('i').unpack(f[0:4])[0]
    if vartype != 0:
        raise ValueError("Only Support dense weight")
    dtype = struct.Struct('i').unpack(f[12:16])[0]
    shape_size = struct.Struct('q').unpack(f[16:24])[0]
    shape = struct.Struct('q' * shape_size).unpack(f[24:24+shape_size * 8])
    init_str = struct.Struct('q').unpack(f[24+shape_size * 8 + 8:24+shape_size * 8 + 16])[0]
    f = f[24 + shape_size * 8 + 16 + init_str:]

    if dtype == 0:
        dtype = np.int8
    elif dtype == 1:
        dtype = np.int16
    elif dtype == 2:
        dtype = np.int32
    elif dtype == 3:
        dtype = np.int64
    elif dtype == 4:
        dtype = np.float32
    elif dtype == 5:
        dtype = np.float64
    else:
        raise ValueError("Dtype Error: %s" % dtype)

    size = 1
    for dim in shape:
        size *= dim
    x = np.frombuffer(f, dtype = dtype, count = size).reshape(shape)
    return x

dict_lock = multiprocessing.Lock()
def _update_save_dict(ckpt, var, save_dict, backend='debug', title='var'):
    var = var.replace('/', '$')
    s = []
    k = 0
    while True:
        f = ''
        fn = os.path.join(ckpt, "%s^%s" % (var, k)).replace("$", "\\$")
        if _is_hdfs(fn):
            ret = _system('hadoop fs -ls %s 1>/dev/null 2>/dev/null' % fn)
            if ret != 0:
                break
            f = _popen("hadoop fs -cat %s" % fn).read()
        else:
            ret = _system('ls %s 1>/dev/null 2>/dev/null' % fn)
            if ret != 0:
                break
            f = _popen("cat %s" % fn).read()
        if f == '':
            break
        try:
            s.append(_read_dense_file(f))
        except (ValueError), x:
            print '    %s is not dense, pass' % var
            return
        else:
            pass
        k += 1
    if len(s) > 0:
        weight = np.concatenate(s)
        global dict_lock
        if backend == 'mx':
            from mxnet import ndarray
            dict_lock.acquire()
            save_dict.update({('%s:%s' % (title, var)) : ndarray.array(weight)})
            dict_lock.release()
            print '    %s export success' % var
        elif backend == 'tf':
            dict_lock.acquire()
            save_dict.update({(var) : (weight)})
            dict_lock.release()
            print '    %s export success' % var
        else:
            print '    %s export success:' % var, weight

def _update_save_dict_run(n, ckpt, var_list, save_dict, backend='debug', title='var'):
    _update_save_dict(ckpt, var_list[n], save_dict, backend, title)

def export_dense_debug(checkpoints_dir, arg_list):
    ckpt = get_latest_ckpt_v2(checkpoints_dir)
    print 'export dense debug from %s' % ckpt    
    save_dict = dict()
    print '# export arg:', arg_list
    for arg in arg_list:
        _update_save_dict(ckpt, arg, save_dict, 'debug')

def export_dense_mx(ckpt_dir, arg_list, aux_list, output_dir):
    ckpt = get_latest_ckpt_v2(ckpt_dir)
    print 'export dense mx from %s' % ckpt    
    manager = multiprocessing.Manager()
    save_dict = manager.dict()
    print '# export arg:', arg_list
    p_list = []
    for n in xrange(len(arg_list)):
        p = multiprocessing.Process(target=_update_save_dict_run, args=(n, ckpt, arg_list, save_dict, 'mx', 'var'))
        p_list.append(p)
        p.start()
    print '# export aux:', aux_list
    for n in xrange(len(aux_list)):
        p = multiprocessing.Process(target=_update_save_dict_run, args=(n, ckpt, aux_list, save_dict, 'mx', 'aux'))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()

    save_dict = dict(save_dict)
    from mxnet import ndarray
    ndarray.save('dense.txt', save_dict)
    output(output_dir, 'dense.txt')
    return save_dict

def export_dense_tf(checkpoints_dir, output_dir):
    ckpt = get_latest_ckpt_v2(checkpoints_dir)
    print 'export dense tf from %s' % ckpt
    if _is_hdfs(ckpt):
        arg_list = _popen("hadoop fs -ls %s |awk -F'/' '{print $NF}' |grep '\$' |grep '\^0'" % ckpt).readlines()
    else:
        arg_list = _popen("ls %s" % ckpt).readlines()
    arg_list = [i[:-3].split(' ')[-1] for i in arg_list]
    
    print '# export tf arg:', arg_list
    manager = multiprocessing.Manager()
    save_dict = manager.dict()
    p_list = []
    for n in xrange(len(arg_list)):
        p = multiprocessing.Process(target=_update_save_dict_run, args=(n, ckpt, arg_list, save_dict, 'tf'))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()

    save_dict = dict(save_dict)
    np.savez('dense', **save_dict)  # * for list, ** for map
    output(output_dir, 'dense.npz')
    return save_dict

def _transfer_sparse_idx_to_hash(fname, dim, offset):
    result = list()
    ori_fname = fname
    if _is_hdfs(ori_fname):
        _cmd('rm -f sparse_var')
        _cmd('hadoop fs -get %s sparse_var' % fname)
        fname = 'sparse_var'
    with open(fname, 'r') as f:
        t = f.read()
        length = len(t)
        idx = 0
        begin = 0
        for i in xrange(length):
            if t[i] == ',':
                idx = idx + 1
                if idx == dim:
                    idx = 0
                    result.append(str(offset) + ',' + t[begin:i] + '\n')
                    offset = offset + 1
                    begin = i + 1
    with open(fname, 'w') as f:
        f.writelines(result)
    if _is_hdfs(ori_fname):
        _cmd('hadoop fs -put -f %s %s' % (fname, ori_fname))

def _export_sparse_var(ckpt_dir, output_dir, var, vtype='hash', dim=18):
    def _string_to_int8(src):
        return np.array([ord(ch) for ch in src], dtype=np.int8)
    print(var)
    op = xdl.ps_convert_ckpt_variable_op(checkpoint_dir=_string_to_int8(ckpt_dir),
                                         output_dir=_string_to_int8(output_dir),
                                         variables=_string_to_int8(var))
    xdl.execute(op)
    if vtype != 'hash':
        _transfer_sparse_idx_to_hash(os.path.join(output_dir, var), dim, offset=0)

def export_sparse(ckpt_dir, output_dir, var_list, vtype='hash', dim=18):
    with open('sparse.txt', 'w') as f:
        for var in var_list:
            f.writelines('%s\n' % var)
    output(output_dir, 'sparse.txt')
    #for var in var_list:
    _export_sparse_var(ckpt_dir, output_dir, (',').join(var_list), vtype, dim)

