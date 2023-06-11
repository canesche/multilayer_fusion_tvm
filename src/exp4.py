import tvm, logging, sys, os, time
from tvm import te, topi, autotvm
import numpy as np

from utils import auto_fusion_schedule_order, get_best_time, p_value

dtype="float32"

@autotvm.template("fusion")
def fusion(N, H, W, CO, CI, KH, KW, stride, padding, order):

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

    cfg = autotvm.get_config()
    auto_fusion_schedule_order(s, cfg, tensors, order)

    return s, args


def normal(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    bias = te.placeholder((1, W, 1), name="bias")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    relu1 = topi.nn.relu(bias1)
    pool1 = topi.nn.pool2d(relu1, kernel=(2,2), stride=(2,2), padding=(0,0,0,0), dilation=(1,1), pool_type="max", layout="NCHW")

    #print(conv1.shape)
    #print(bias1.shape)
    #print(relu1.shape)
    #print(pool1.shape)

    s = te.create_schedule(pool1.op)
    args = [data, kernel, bias]

    return s, args


def execute_normal(N, H, W, CO, CI, KH, KW, stride, padding, dev, target):

    s, args = normal(N, H, W, CO, CI, KH, KW, stride, padding)

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
    b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev) 

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
        runner=autotvm.LocalRunner(repeat=5, number=10, min_repeat_ms=100),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    
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
        dev = tvm.cpu(0)
        target = "llvm"
    elif arch == "arm":
        dev = tvm.cpu(0)
        target = "llvm -device=arm_cpu"
    elif arch == "cuda":
        target = "cuda"
        dev = tvm.cuda(0)
    else:
        print("archictecture undefined!")
        exit(0)
    
    for l, i in enumerate(interval):
        H, W = (i, i)
        print("\n(%d,%d)" %(i,i))
        r_normal = execute_normal(N, H, W, CO, CI, KH, KW, stride, padding, dev, target)
        for j in range(len(order)):
            r = execute_autoTVM("fusion", fusion, N, H, W, CO, CI, KH, KW, stride, padding, order[j], j, dev, target)
            print(p_value(r_normal, r))
