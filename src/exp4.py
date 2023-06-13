import tvm, logging, sys, os, time
from tvm import te, topi, autotvm
import numpy as np

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

from utils import auto_fusion_schedule_order, get_best_time, p_value

dtype="float32"

@autotvm.template("fusion")
def fusion(N, H, W, CO, CI, KH, KW, stride, padding, order):

    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    bias = te.placeholder((1, W, 1), name="bias", dtype=dtype)
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel", dtype=dtype)

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    relu1 = topi.nn.relu(bias1)
    pool1 = topi.nn.pool2d(relu1, kernel=(2,2), stride=(2,2), padding=(0,0,0,0), dilation=(1,1), pool_type="max", layout="NCHW")

    s = te.create_schedule(pool1.op)
    args = [data, kernel, bias]
    tensors = [conv1, bias1, relu1, pool1]

    cfg = autotvm.get_config()
    auto_fusion_schedule_order(s, cfg, tensors, order)

    return s, args


def normal(N, H, W, CO, CI, KH, KW, stride, padding, arch):

    data = te.placeholder((N, CI, H, W), name="data")
    bias = te.placeholder((1, W, 1), name="bias")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    relu1 = topi.nn.relu(bias1)
    pool1 = topi.nn.pool2d(relu1, kernel=(2,2), stride=(2,2), padding=(0,0,0,0), dilation=(1,1), pool_type="max", layout="NCHW")

    s = te.create_schedule(pool1.op)
    args = [data, kernel, bias]
    tensors = [conv1, bias1, relu1, pool1]
    #tensors = [conv1]

    if arch == "cuda":
        tile = 4
        num_thread = 4
        vthread = 2
        block_factor = tile * num_thread

        conv_global = s.cache_write(conv1, "global")
        #s.compute_at(conv_global, )
        bias_global = s.cache_write(bias1, "global")

        #O1 = s.cache_write(conv1, "local")
        #OL = s.cache_write(pool1, "local")

        # create cache stage
        #AA = s.cache_read(data, "shared", [O1])
        #WW = s.cache_read(kernel, "shared", [OL])
        #BB = s.cache_read(bias, "shared", [OL])

        for t in tensors:
            # Get the GPU thread indices
            block_x = te.thread_axis("blockIdx.x")
            block_y = te.thread_axis("blockIdx.y")
            #block_z = te.thread_axis("blockIdx.z")
            thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
            thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
            #thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
            #thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")
            
            axis = s[t].op.axis
            #bz = s[t].fuse(axis[0], axis[1])
            by, fi = s[t].split(axis[2], factor=block_factor)
            bx, ni = s[t].split(axis[3], factor=block_factor)

            #s[t].bind(bz, block_z)
            s[t].bind(by, block_y)
            s[t].bind(bx, block_x)

            #tyz, fi = s[t].split(fi, nparts=vthread)  # virtual thread split
            #txz, ni = s[t].split(ni, nparts=vthread)  # virtual thread split
            ty, fi = s[t].split(fi, nparts=num_thread)
            tx, ni = s[t].split(ni, nparts=num_thread)
            #s[t].reorder( by, bx, ty, tx, fi, ni)

            #s[t].bind(tyz, thread_yz)
            #s[t].bind(txz, thread_xz)
            s[t].bind(ty, thread_y)
            s[t].bind(tx, thread_x)
            #break

    return s, args


def execute_normal(N, H, W, CO, CI, KH, KW, stride, padding, dev, target, arch):

    s, args = normal(N, H, W, CO, CI, KH, KW, stride, padding, arch)

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
    b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev) 

    with tvm.target.Target(target):
        with tvm.transform.PassContext(opt_level=0):
            mod = tvm.build(s, args=args, target=target, name="main")
            mod(d_tvm, f_tvm, b_tvm)
    r = []
    for _ in range(5):
        evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
        mean_time = evaluator(d_tvm, f_tvm, b_tvm)
        r.append(mean_time.mean)
    r = np.array(r)
    print("%s,%.6f,%.6f" %("normal", np.mean(r), np.std(r)))
    return r

def execute_autoTVM(tag_name, func, N, H, W, CO, CI, KH, KW, stride, padding, order, number, dev, target):

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
    b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev) 

    task = autotvm.task.create(tag_name, args=(N, H, W, CO, CI, KH, KW, stride, padding, order), target=target)
    #print(task.config_space)

    space_size = len(task.config_space)
    
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=5, number=10, min_repeat_ms=100, enable_cpu_cache_flush=True, timeout=100),
    )
    tuner = autotvm.tuner.GridSearchTuner(task)
    
    record_file = "../results/exp4_%s_%dx%d_%d.log" %(tag_name, H, W, number)
    
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
                s, args = func(N, H, W, CO, CI, KH, KW, stride, padding, order)
                mod = tvm.build(s, args=args, target=target, name="main")
                mod(d_tvm, f_tvm, b_tvm)
        r = []
        for _ in range(5):
            evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
            mean_time = evaluator(d_tvm, f_tvm, b_tvm)
            r.append(mean_time.mean)
        r_measured = np.array(r)
    
    r, conf = get_best_time(record_file)

    print("%s,%.6f,%.6f,%.6f,%.6f,%.4f" %(tag_name, np.mean(r), np.std(r), np.mean(r_measured), np.std(r_measured), final_time))    
    print(conf)
    
    return r

if __name__ == "__main__":

    N = 1
    CO, CI = (3, 3)
    KH, KW = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    interval = [128, 256, 512, 1024, 2048]
    order = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]

    arch = "x86"
    if len(sys.argv) > 1:
        arch = sys.argv[1]

    if arch == "x86":
        target = "llvm"
        dev = tvm.cpu(0)
    elif arch == "arm":
        target = "llvm -device=arm_cpu"
        dev = tvm.cpu(0)
    elif arch == "cuda":
        target = "cuda"
        dev = tvm.cuda(0)
    else:
        print("archictecture undefined!")
        exit(0)
    
    for l, i in enumerate(interval):
        H, W = (i, i)
        print("\n(%d,%d)" %(i,i))
        r_normal = execute_normal(N, H, W, CO, CI, KH, KW, stride, padding, dev, target, arch)

        for j in range(len(order)):
            r = execute_autoTVM("fusion", fusion, N, H, W, CO, CI, KH, KW, stride, padding, order[j], j, dev, target)
            print(p_value(r_normal, r))