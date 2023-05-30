import tvm, sys, os, time
from tvm import te, topi, autotvm, relay
import numpy as np

from utils import auto_fusion_schedule, auto_tile_schedule, get_best_time, p_value

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

dtype="float32"
dev = tvm.cpu(0)


def normal(N):

    A = te.compute((N,), lambda i: i + 1, name="A")
    B = te.compute((N//2, ), lambda i: A[i*2] + i, name="B")

    s = te.create_schedule(B)
    args = []

    return s, args


@autotvm.template("tile")
def tile(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    relu1 = topi.nn.relu(conv1)

    s = te.create_schedule(relu1.op)
    args = [data, kernel]
    tensors = [conv1, relu1]

    search_space = [0,1,2,4,8,12,16,20,24,28,32]

    cfg = autotvm.get_config()
    auto_tile_schedule(s, cfg, tensors, search_space)

    return s, args


@autotvm.template("fusion_tile")
def fusion_tile(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    relu1 = topi.nn.relu(conv1)

    s = te.create_schedule(relu1.op)
    args = [data, kernel]
    tensors = [conv1, relu1]

    search_space = [0,1,2,4,8,12,16,20,24,28,32]

    cfg = autotvm.get_config()
    auto_fusion_schedule(s, cfg, tensors)
    auto_tile_schedule(s, cfg, tensors, search_space)

    return s, args


@autotvm.template("fusion")
def fusion(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    relu1 = topi.nn.relu(conv1)

    s = te.create_schedule(relu1.op)
    args = [data, kernel]
    tensors = [conv1, relu1]

    cfg = autotvm.get_config()
    auto_fusion_schedule(s, cfg, tensors)

    return s, args


def execute_normal(N):
    s, args = normal(N)

    with tvm.transform.PassContext(opt_level=0):
        mod = tvm.build(s, args=args, target="llvm", name="main")
        mod()
    r = []
    for _ in range(5):
        evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
        mean_time = evaluator()
        r.append(mean_time.mean)
    print("%s,%.6f,%.6f" %("normal", np.mean(r), np.std(r)))
    return r


def execute_autoTVM(tag_name, func, N, H, W, CO, CI, KH, KW, stride, padding):

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)

    task = autotvm.task.create(tag_name, args=(N, H, W, CO, CI, KH, KW, stride, padding), target="llvm")
    #print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=7, number=10, min_repeat_ms=100, enable_cpu_cache_flush=True),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    record_file = "../results/exp1_%s_%dx%d.log" %(tag_name, H, W)
    
    if os.path.isfile(record_file):
        os.remove(record_file)

    start = time.time()
    tuner.tune(
        n_trial=200,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(record_file)],
    )
    final_time = time.time() - start

    with tvm.target.Target("llvm"):
        with tvm.transform.PassContext(opt_level=3):
            with autotvm.apply_history_best(record_file):
                s, args = func(N, H, W, CO, CI, KH, KW, stride, padding)
                mod = tvm.build(s, args=args, target="llvm", name="main")
                mod(d_tvm, f_tvm)
        r = []
        for _ in range(5):
            evaluator = mod.time_evaluator(mod.entry_name, dev, number=10, repeat=1)
            mean_time = evaluator(d_tvm, f_tvm)
            r.append(mean_time.mean)
        r_measured = np.array(r)
    r, conf = get_best_time(record_file, False)

    print("%s,%.6f,%.6f,%.6f,%.6f,%.2f" %(tag_name, np.mean(r), np.std(r), np.mean(r_measured), np.std(r_measured), final_time), end=",")    
    #print(conf)
    
    return r

if __name__ == "__main__":

    interval = [512]

    for i in interval:
        print("\n(%d)" %(i))

        r_normal = execute_normal(i)
        #r_fusion = execute_autoTVM("fusion", fusion, N, H, W, CO, CI, KH, KW, stride, padding)
        #print("%.6f" %(p_value(r_normal, r_fusion)))
        #r_tile = execute_autoTVM("tile", tile, N, H, W, CO, CI, KH, KW, stride, padding)
        #print("%.6f" %(p_value(r_normal, r_tile)))
        #r_fusion_tile = execute_autoTVM("fusion_tile", fusion_tile, N, H, W, CO, CI, KH, KW, stride, padding)
        #print("%.6f" %(p_value(r_normal, r_fusion_tile)))