import tvm, logging, sys, os
from tvm import te, topi, autotvm, relay
import numpy as np

from utils import auto_fusion_schedule, auto_tile_schedule, get_best_time, p_value

#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

dtype="float32"
dev = tvm.cpu(0)


def normal(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    relu1 = topi.nn.relu(conv1)

    s = te.create_schedule(relu1.op)
    args = [data, kernel]

    return s, args


@autotvm.template("only_opt")
def only_opt(N, H, W, CO, CI, KH, KW, stride, padding):

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


@autotvm.template("fusion_opt")
def fusion_opt(N, H, W, CO, CI, KH, KW, stride, padding):

    # Create the input tensor
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    relu1 = topi.nn.relu(conv1)

    # Create a TVM schedule for the computation
    s = te.create_schedule(relu1.op)
    args = [data, kernel]
    tensors = [conv1, relu1]

    search_space = [0,1,4,8,12,16,20,24,28,32]

    cfg = autotvm.get_config()

    # merging the kernel first and then applying tile optimization works!
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


def execute_normal(N, H, W, CO, CI, KH, KW, stride, padding):
    s, args = normal(N, H, W, CO, CI, KH, KW, stride, padding)

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
    b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev)

    with tvm.transform.PassContext(opt_level=0):
        mod = tvm.build(s, args=args, target="llvm", name="main")
        mod(d_tvm, f_tvm, b_tvm)
    r = []
    for _ in range(5):
        evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
        mean_time = evaluator(d_tvm, f_tvm, b_tvm)
        r.append(mean_time.mean)
    r = np.array(r)
    print("%s,%.6f,%.6f" %("normal", np.mean(r), np.std(r)))
    return r


def execute_autoTVM(tag_name, func, N, H, W, CO, CI, KH, KW, stride, padding):

    d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
    f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
    b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev)

    task = autotvm.task.create(tag_name, args=(N, H, W, CO, CI, KH, KW, stride, padding), target="llvm")
    #print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=5, number=20, min_repeat_ms=100, enable_cpu_cache_flush=True),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    record_file = "exp1_%s_%dx%d.log" %(tag_name, H, W)
    
    if os.path.isfile(record_file):
        os.remove(record_file)

    tuner.tune(
        n_trial=200,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(record_file)],
    )

    #with tvm.target.Target("llvm"):
    #    with tvm.transform.PassContext(opt_level=3):
    #        with autotvm.apply_history_best(record_file):
    #            s, args = func(N, H, W, CO, CI, KH, KW, stride, padding)
    #            mod = tvm.build(s, args=args, target="llvm", name="main")
    #            mod(d_tvm, f_tvm, b_tvm)
    #    r = []
    #    for _ in range(3):
    #        evaluator = mod.time_evaluator(mod.entry_name, dev, number=20, repeat=1)
    #        mean_time = evaluator(d_tvm, f_tvm, b_tvm)
    #        r.append(mean_time.mean)
    #    r = np.array(r)
    #    print("%s,%.6f,%.6f" %(tag_name, np.mean(r), np.std(r)), end=",")
    
    r, conf = get_best_time(record_file, False)

    print("%s,%.6f,%.6f" %(tag_name, np.mean(r), np.std(r)), end=",")    
    print(conf)
    
    return r

if __name__ == "__main__":

    N = 1
    CO, CI = (3, 3)
    KH, KW = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    interval = [128, 256, 512, 1024, 2048]

    for i in interval:
        H, W = (i, i)
        print("\n(%d,%d)" %(i,i))

        #r_normal = execute_normal(N, H, W, CO, CI, KH, KW, stride, padding)
        #r_fusion = execute_autoTVM("fusion", fusion, N, H, W, CO, CI, KH, KW, stride, padding)
        r_only_opt = execute_autoTVM("only_opt", only_opt, N, H, W, CO, CI, KH, KW, stride, padding)
        #r_fusion_opt = execute_autoTVM("fusion_opt", fusion_opt, N, H, W, CO, CI, KH, KW, stride, padding)

        #print(p_value(r_normal, r_fusion))
        #print(p_value(r_normal, r_only_opt))
        #print(p_value(r_normal, r_fusion_opt))