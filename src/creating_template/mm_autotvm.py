import sys, os
import numpy as np
import time
import tvm
from tvm import te, autotvm
from module.creating_template import *
from utils import get_best_time

@autotvm.template("mm")
def matmul(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)
    args = [A, B, C]
    tensors = [C]

    cfg = autotvm.get_config()
    cfg.define_knob("SP", [i for i in range(6)])

    axis = s[C].op.axis
    reduce_axis = s[C].op.reduce_axis

    if cfg["SP"].val != 0:
        _, _ = s[C].split(axis[0], cfg["SP"].val)

    #cfg = Template_ansor(s, tensors)
    #cfg.space('SP', [2, 0, 1000, [20, 1, 2], 1])

    #print(cfg.cfg)

    return [s, args]

if __name__ == "__main__":

    arch = "cpu"

    if len(sys.argv) == 1:
        arch = sys.argv[1]

    if arch == "cpu":
        target = tvm.target.Target("llvm")
        dev = tvm.cpu() 
    elif arch == "cuda":
        target = tvm.target.Target("cuda")
        dev = tvm.cuda()
    else:
        print("Archtecture doesn't support.")
        exit(0)

    print("Arch:", arch)

    ## Create the search task
    
    N, L, M = 1000, 800, 700
    dtype = "float32"

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    
    task = autotvm.task.create("mm", args=(N, L, M, dtype), target=target)
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, number=10, min_repeat_ms=100),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    record_file = "testing.log"
    
    if os.path.isfile(record_file):
        os.remove(record_file)

    start = time.time()
    tuner.tune(
        n_trial=200,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(record_file)],
    )
    final_time = time.time() - start

    with tvm.target.Target(target):
        with tvm.transform.PassContext(opt_level=3):
            with autotvm.apply_history_best(record_file):
                s, args = matmul(N, L, M)
                mod = tvm.build(s, args=args, name="main")
                mod(a_tvm, b_tvm, c_tvm)

    '''
    ## Check correctness and evaluate performance
    with auto_scheduler.ApplyHistoryBest(log_file):
        func = tvm.build(sch, args, target)
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        func(a_tvm, b_tvm, c_tvm)

    # Check results
    #np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time.
    #evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
    #eval = evaluator(a_tvm, b_tvm, c_tvm)
    #print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

    #print("Equivalent python schedule:")
    #print(task.print_best(log_file))
    '''

    time_avg, best_cfg = get_best_time(record_file)

    print("Time spent:", time_avg)
    print("Config:", best_cfg) 