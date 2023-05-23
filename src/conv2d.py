import tvm
from tvm import te, topi
import numpy as np

dtype="float32"
dev = tvm.cpu(0)

# Input shape
batch_size = 1
in_channels = 3
in_size = 256

# Filter shape
out_channels = 64
filter = 3

# Strides and padding
stride = 1
padding = 1

i_shape = (batch_size, in_channels, in_size, in_size)
w_shape = (out_channels, in_channels, filter, filter)
b_shape = (1, in_size, 1)

def test():

    # Create the input tensor
    input = te.placeholder(i_shape, name='input')
    filter = te.placeholder(w_shape, name='weight')
    bias = te.placeholder(b_shape, name='bias')

    # Perform the convolution using topi
    conv1 = topi.nn.conv2d(input, filter, strides=(stride, stride), padding=(padding, padding), dilation=1, data_layout="NCHW")
    bias1 = topi.add(conv1, bias)
    #relu1 = topi.nn.relu(bias1)

    # Create a TVM schedule for the computation
    s = te.create_schedule(bias1.op)
    args = [input, filter, bias]
    print(tvm.lower(s, args, simple_mode=True))

    s[conv1].compute_at(s[bias1], bias1.op.axis[0])
    s[conv1].compute_at(s[bias1], bias1.op.axis[1])
    s[conv1].compute_at(s[bias1], bias1.op.axis[2])
    s[conv1].compute_at(s[bias1], bias1.op.axis[3])

    #s[bias1].compute_at(s[relu1], relu1.op.axis[0])
    #s[bias1].compute_at(s[relu1], relu1.op.axis[1])
    #s[bias1].compute_at(s[relu1], relu1.op.axis[2])
    #s[bias1].compute_at(s[relu1], relu1.op.axis[3])

    print(conv1)
    print("conv1:", conv1.op.axis)
    print("bias1:", bias1.op.axis)
    #print("relu1:", relu1.op.axis)

    print(tvm.lower(s, args, simple_mode=True))

    return s, args

# Build the TVM module
s, args = test()
mod = tvm.build(s, args=args, target="llvm", name="main")

d_tvm = tvm.nd.array((np.random.uniform(size=i_shape)).astype(dtype), device=dev)
f_tvm = tvm.nd.array((np.random.uniform(size=w_shape)).astype(dtype), device=dev)
b_tvm = tvm.nd.array((np.random.uniform(size=b_shape)).astype(dtype), device=dev)

# Print the generated code
#print(mod.get_source())

mod(d_tvm, f_tvm, b_tvm)

evaluator = mod.time_evaluator(mod.entry_name, dev, number=10, repeat=3)
mean_time = evaluator(d_tvm, f_tvm, b_tvm)

print(mean_time)