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

    print(tvm.lower(s, args=args))

    #s[conv1].compute_at(s[bias1], bias1.op.axis[3])
    #s[bias1].compute_at(s[relu1], relu1.op.axis[3])
    #s[relu1].compute_at(s[pool1], pool1.op.axis[3])
    #s[bias1].compute_inline()
    #s[relu1].compute_inline()
    
    s[relu1].compute_at(s[pool1], pool1.op.axis[3])
    s[bias1].compute_at(s[relu1], relu1.op.axis[3])
    s[conv1].compute_at(s[bias1], bias1.op.axis[3])

    print(tvm.lower(s, args=args))
    print()

    #s[conv1].compute_at(s[bias1], bias1.op.axis[2])

    #print(tvm.lower(s, args=args))
    #print()

    #s[conv1].compute_at(s[bias1], bias1.op.axis[1])
    #s[conv1].compute_at(s[bias1], bias1.op.axis[2])

    #print(tvm.lower(s, args=args))

    #cfg = autotvm.get_config()
    #auto_fusion_schedule_order(s, cfg, tensors, order)

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
    
    return r

if __name__ == "__main__":

    N = 1
    CO, CI = (3, 3)
    KH, KW = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    interval = [128]
    #order = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    order = [[0,1,2]]

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
        #r_normal = execute_normal(N, H, W, CO, CI, KH, KW, stride, padding, dev, target)
        
        r =[]
        for j in range(len(order)):
            r.append(execute_autoTVM("fusion", fusion, N, H, W, CO, CI, KH, KW, stride, padding, order[j], j, dev, target))
