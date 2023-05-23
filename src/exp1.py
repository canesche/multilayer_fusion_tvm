import tvm, logging, sys
from tvm import te, topi, autotvm, relay
import numpy as np

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

dtype="float32"
dev = tvm.cpu(0)

@autotvm.template("without_fusion")
def without_fusion(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, W, 1), name='bias')

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    relu1 = topi.nn.relu(bias1)

    # Create a TVM schedule for the computation
    s = te.create_schedule(relu1.op)
    args = [data, kernel, bias]

    n, f, y, x = s[conv1].op.axis
    rc, ry, rx = s[conv1].op.reduce_axis
   
    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    pad_data = s[conv1].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    s[conv1].pragma(n, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[conv1].pragma(n, "unroll_explicit", cfg["unroll_explicit"].val)

    return s, args

@autotvm.template("with_fusion")
def with_fusion(N, H, W, CO, CI, KH, KW, stride, padding):

    # Create the input tensor
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, W, 1), name='bias')

    conv1 = topi.nn.conv2d(data, kernel, strides=stride, padding=padding, dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    relu1 = topi.nn.relu(bias1)

    # Create a TVM schedule for the computation
    s = te.create_schedule(relu1.op)
    args = [data, kernel, bias]
    #print(tvm.lower(s, args, simple_mode=True))

    s[conv1].compute_at(s[bias1], bias1.op.axis[0])
    s[conv1].compute_at(s[bias1], bias1.op.axis[1])
    s[conv1].compute_at(s[bias1], bias1.op.axis[2])
    s[conv1].compute_at(s[bias1], bias1.op.axis[3])

    s[bias1].compute_at(s[relu1], relu1.op.axis[0])
    s[bias1].compute_at(s[relu1], relu1.op.axis[1])
    s[bias1].compute_at(s[relu1], relu1.op.axis[2])
    s[bias1].compute_at(s[relu1], relu1.op.axis[3])

    #print(tvm.lower(s, args, simple_mode=True))

    #print("conv1:", conv1.op.axis)
    #print("red1:", conv1.op.reduce_axis)
    #print("bias1:", bias1.op.axis)
    #print("relu1:", relu1.op.axis)

    return s, args


N, H, W, CO, CI, KH, KW, strides, padding = 1, 3, 3, 512, 512, 3, 3, (1, 1), (1, 1)

task = autotvm.task.create("with_fusion", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="llvm")
print(task.config_space)

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

tuner = autotvm.tuner.XGBTuner(task)
record_file = "conv2d.log"
tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file(record_file)],
)

dispatch_context = autotvm.apply_history_best(record_file)
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:", best_config)

d_tvm = tvm.nd.array((np.random.uniform(size=(N, CI, H, W))).astype(dtype), device=dev)
f_tvm = tvm.nd.array((np.random.uniform(size=(CO, CI, KH, KW))).astype(dtype), device=dev)
b_tvm = tvm.nd.array((np.random.uniform(size=(1, W, 1))).astype(dtype), device=dev)

with autotvm.apply_history_best(record_file):
    with tvm.target.Target("llvm"):
        # Build the TVM module
        s, args = with_fusion(N, H, W, CO, CI, KH, KW, strides, padding)
        mod = tvm.build(s, args=args, target="llvm", name="main")
        print(tvm.lower(s, args, simple_mode=True))

mod(d_tvm, f_tvm, b_tvm)

evaluator = mod.time_evaluator(mod.entry_name, dev, number=10, repeat=3)
mean_time = evaluator(d_tvm, f_tvm, b_tvm)

print(mean_time)