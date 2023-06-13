import tvm, os, time
from tvm import te, topi, relay, autotvm
from tvm.contrib import graph_executor, graph_runtime
import numpy as np

import logging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("autotvm")

from utils import auto_fusion_schedule_seq, get_best_time, p_value


def resnet_18(input_shape, num_classes, data_layout='NCHW'):

    # Convolutional helper function
    def conv2d(data, in_channel, out_channel, kernel, stride, padding, name):
        w1 = topi.full_like(te.placeholder((out_channel, in_channel, kernel, kernel), name='weight'), 0.01)
        #topi.reshape(weight, (out_channel, in_channel, kernel, kernel))
        b1 = topi.full_like(te.placeholder((1,out_channel,1,1), name='bias'), 0.01)
        #topi.reshape(bias, (1, out_channel, 1, 1))
        conv = topi.nn.conv2d(data, w1, stride, padding, dilation=(1,1), data_layout=data_layout)
        bias_add = topi.add(conv, b1)
        return bias_add

    # Basic block helper function
    def basic_block(data, in_channel, out_channel, stride, name):
        identity = data
        out = conv2d(data, in_channel, out_channel, 3, stride, 1, name=name+'_conv1')
        out = topi.nn.relu(out)
        out = conv2d(out, out_channel, out_channel, 3, 1, 1, name=name+'_conv2')
        if in_channel != out_channel or stride != 1:
            identity = conv2d(out, in_channel, out_channel, 1, stride, 0, name=name+'_downsample')
        identity = topi.reshape(identity, out.shape)
        out = topi.add(out, identity)
        out = topi.nn.relu(out)
        return out

    # Define input tensor
    data = te.placeholder(input_shape, name='data')
    
    # Stage 1
    conv1 = conv2d(data, 3, 64, 7, 2, 3, name='stage1')
    relu1 = topi.nn.relu(conv1)
    pool1 = topi.nn.pool2d(relu1, kernel=(3,3), stride=(2,2), dilation=(1,1), padding=(1,1,1,1), layout="NCHW", pool_type="max")
    
    # Stage 2
    bb2_1 = basic_block(pool1, 64, 64, 1, name='stage2_1')
    bb2_2 = basic_block(bb2_1, 64, 64, 1, name='stage2_2')

    # Stage 3
    bb3_1 = basic_block(bb2_2, 64, 128, 2, name='stage3_1')
    bb3_2 = basic_block(bb3_1, 128, 128, 1, name='stage3_2')

    # Stage 4
    bb4_1 = basic_block(bb3_2, 128, 256, 2, name='stage4_1')
    bb4_2 = basic_block(bb4_1, 256, 256, 1, name='stage4_2')

    # Stage 5
    bb5_1 = basic_block(bb4_2, 256, 512, 2, name='stage5_1')
    bb5_2 = basic_block(bb5_1, 512, 512, 1, name='stage5_2')

    # Global average pooling
    g_avg_pool = topi.nn.global_pool(bb5_2, "avg", layout=data_layout)

    # Fully connected layer
    w_fc = topi.full_like(te.placeholder((num_classes, 512), name='fc_weight'), 0.01)
    #topi.reshape(weight, (num_classes, 512)) #placeholder((num_classes, 512), name='fc_weight')
    b_fc = topi.full_like(te.placeholder((num_classes,), name='fc_bias'), 0.01)
    #topi.reshape(bias, (num_classes,)) #te.placeholder((num_classes,), name='fc_bias')
    flat = topi.nn.flatten(g_avg_pool)
    dense = topi.nn.dense(flat, w_fc)
    bias_f = topi.add(dense, b_fc)

    # Softmax
    out_soft = topi.nn.softmax(bias_f)

    # Generate schedule
    s = te.create_schedule([out_soft.op])
    args = [data]
    tensors = [conv1, relu1, pool1, bb2_1, bb2_2, bb3_1, 
               bb3_2, bb4_1, bb4_2, bb5_1, bb5_2, g_avg_pool, 
               flat, dense, bias_f, out_soft]

    print(tvm.lower(s, [data], simple_mode=True))

    s[out_soft].compute_inline()
    print()
    print(tvm.lower(s, [data], simple_mode=True))
    return s, args

@autotvm.template("fusion")
def resnet_18_fusion(input_shape, num_classes, data_layout='NCHW'):

    # Convolutional helper function
    def conv2d(data, in_channel, out_channel, kernel, stride, padding, name):
        w1 = topi.full_like(te.placeholder((out_channel, in_channel, kernel, kernel), name='weight'), 0.01)
        #topi.reshape(weight, (out_channel, in_channel, kernel, kernel))
        b1 = topi.full_like(te.placeholder((1,out_channel,1,1), name='bias'), 0.01)
        #topi.reshape(bias, (1, out_channel, 1, 1))
        conv = topi.nn.conv2d(data, w1, stride, padding, dilation=(1,1), data_layout=data_layout)
        bias_add = topi.add(conv, b1)
        return bias_add

    # Basic block helper function
    def basic_block(data, in_channel, out_channel, stride, name):
        identity = data
        out = conv2d(data, in_channel, out_channel, 3, stride, 1, name=name+'_conv1')
        out = topi.nn.relu(out)
        out = conv2d(out, out_channel, out_channel, 3, 1, 1, name=name+'_conv2')
        if in_channel != out_channel or stride != 1:
            identity = conv2d(out, in_channel, out_channel, 1, stride, 0, name=name+'_downsample')
        identity = topi.reshape(identity, out.shape)
        out = topi.add(out, identity)
        out = topi.nn.relu(out)
        return out

    # Define input tensor
    data = te.placeholder(input_shape, name='data')
    
    # Stage 1
    conv1 = conv2d(data, 3, 64, 7, 2, 3, name='stage1')
    relu1 = topi.nn.relu(conv1)
    pool1 = topi.nn.pool2d(relu1, kernel=(3,3), stride=(2,2), dilation=(1,1), padding=(1,1,1,1), layout="NCHW", pool_type="max")

    print(conv1.shape)
    print(relu1.shape)
    print(pool1.shape)
    
    # Stage 2
    bb2_1 = basic_block(pool1, 64, 64, 1, name='stage2_1')
    bb2_2 = basic_block(bb2_1, 64, 64, 1, name='stage2_2')

    '''
    # Stage 3
    bb3_1 = basic_block(bb2_2, 64, 128, 2, name='stage3_1')
    bb3_2 = basic_block(bb3_1, 128, 128, 1, name='stage3_2')

    # Stage 4
    bb4_1 = basic_block(bb3_2, 128, 256, 2, name='stage4_1')
    bb4_2 = basic_block(bb4_1, 256, 256, 1, name='stage4_2')

    # Stage 5
    bb5_1 = basic_block(bb4_2, 256, 512, 2, name='stage5_1')
    bb5_2 = basic_block(bb5_1, 512, 512, 1, name='stage5_2')

    # Global average pooling
    g_avg_pool = topi.nn.global_pool(bb5_2, "avg", layout=data_layout)

    # Fully connected layer
    w_fc = topi.full_like(te.placeholder((num_classes, 512), name='fc_weight'), 0.01)
    #topi.reshape(weight, (num_classes, 512)) #placeholder((num_classes, 512), name='fc_weight')
    b_fc = topi.full_like(te.placeholder((num_classes,), name='fc_bias'), 0.01)
    #topi.reshape(bias, (num_classes,)) #te.placeholder((num_classes,), name='fc_bias')
    flat = topi.nn.flatten(g_avg_pool)
    dense = topi.nn.dense(flat, w_fc)
    bias_f = topi.add(dense, b_fc)

    # Softmax
    out_soft = topi.nn.softmax(bias_f)
    '''

    # Generate schedule
    s = te.create_schedule([bb2_2.op])
    args = [data]
    #tensors = [conv1, relu1, pool1, bb2_1, bb2_2, bb3_1, 
    #           bb3_2, bb4_1, bb4_2, bb5_1, bb5_2, g_avg_pool, 
    #           flat, dense, bias_f, out_soft]
    tensors = [bb2_1, bb2_2]
    
    print(tvm.lower(s, [data], simple_mode=True))

    cfg = autotvm.get_config()
    auto_fusion_schedule_seq(s, cfg, tensors)

    #s[bias_f].compute_inline()

    #print(tvm.lower(s, [data, weight, bias], simple_mode=True))
    return s, args


def execute_normal(input_shape, num_classes, target):

    s, args = resnet_18(input_shape, num_classes, data_layout='NCHW')
    
    d_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype), device=dev)
    
    with tvm.target.Target(target):
        with tvm.transform.PassContext(opt_level=0):
            mod = tvm.build(s, args=args, target=target, name="main")
            mod(d_tvm)
    r = []
    for _ in range(5):
        evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
        mean_time = evaluator(d_tvm)
        r.append(mean_time.mean)
    r = np.array(r)
    print("%s,%.6f,%.6f" %("normal", np.mean(r), np.std(r)))
    return r

def execute_autoTVM(tag_name, func, input_shape, num_classes, target, dev, dtype):

    d_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype), device=dev)

    task = autotvm.task.create(tag_name, args=(input_shape, num_classes), target=target)

    space_size = len(task.config_space)
    
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=1, number=1, enable_cpu_cache_flush=True, timeout=200),
    )
    tuner = autotvm.tuner.GridSearchTuner(task)
    
    record_file = "../results/resnet-18_%s.log" %(tag_name)
    
    if os.path.isfile(record_file):
        os.remove(record_file)

    begin = time.time()
    tuner.tune(
        n_trial=space_size,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(record_file)],
    )
    final_time = time.time() - begin

    with tvm.target.Target(target):
        with tvm.transform.PassContext(opt_level=3):
            with autotvm.apply_history_best(record_file):
                s, args = func(input_shape, num_classes)
                mod = tvm.build(s, args=args, target=target, name="main")
                mod(d_tvm)
        r = []
        for _ in range(5):
            evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
            mean_time = evaluator(d_tvm)
            r.append(mean_time.mean)
        r_measured = np.array(r)
    
    r, conf = get_best_time(record_file)

    print("%s,%.6f,%.6f,%.6f,%.6f,%.4f" %(tag_name, np.mean(r), np.std(r), np.mean(r_measured), np.std(r_measured), final_time))    
    print(conf)
    
    return r


if __name__ == "__main__":

    dev = tvm.cpu()
    target = "llvm"

    input_shape = (1, 3, 224, 224)  # Input shape (batch_size, channels, height, width)
    num_classes = 2  # Number of output classes
    dtype = "float32"

    #r_normal = execute_normal(input_shape, num_classes, target)
    r_fusion = execute_autoTVM("fusion", resnet_18_fusion, input_shape, num_classes, target, dev, dtype)

    #print(p_value(r_normal, r_fusion))