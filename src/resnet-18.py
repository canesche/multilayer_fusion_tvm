import tvm
from tvm import te, topi, relay
from tvm.contrib import graph_executor, graph_runtime
import numpy as np

import logging 
logging.basicConfig(level=logging.DEBUG)

# Define the ResNet-18 architecture using TE
def resnet_18(input_shape, num_classes, data_layout='NCHW'):
    
    #kernel = 3
    #in_channel = 224
    #out_channel = 224
    #weight = te.placeholder((out_channel, in_channel, kernel, kernel), name='weight')
    #bias = te.placeholder((1,out_channel,1,1), name='bias')

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
    out = topi.nn.pool2d(relu1, kernel=(3,3), stride=(2,2), dilation=(1,1), padding=(1,1,1,1), layout="NCHW", pool_type="max")
    
    # Stage 2
    out = basic_block(out, 64, 64, 1, name='stage2_1')
    out = basic_block(out, 64, 64, 1, name='stage2_2')

    # Stage 3
    out = basic_block(out, 64, 128, 2, name='stage3_1')
    out = basic_block(out, 128, 128, 1, name='stage3_2')

    # Stage 4
    out = basic_block(out, 128, 256, 2, name='stage4_1')
    out = basic_block(out, 256, 256, 1, name='stage4_2')

    # Stage 5
    out = basic_block(out, 256, 512, 2, name='stage5_1')
    out = basic_block(out, 512, 512, 1, name='stage5_2')

    # Global average pooling
    g_avg_pool = topi.nn.global_pool(out, "avg", layout=data_layout)

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

    print(tvm.lower(s, [data], simple_mode=True))

    #s[out_soft].compute_inline()

    #print(tvm.lower(s, [data, weight, bias], simple_mode=True))

    # Return build the function
    return tvm.build(s, [data], "llvm", name="resnet-18")


# Example 
dev = tvm.cpu()
input_shape = (1, 3, 224, 224)  # Input shape (batch_size, channels, height, width)
num_classes = 1000  # Number of output classes

lib = resnet_18(input_shape, num_classes)
lib.export_library("compiled_lib.so")
lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")

# Generate random inputs
data_np = np.random.rand(*input_shape).astype("float32")
#weight_fc_np = np.random.rand(num_classes, 512).astype("float32")
#bias_fc_np = np.random.rand(num_classes).astype("float32")

# Create TVM runtime module
module = graph_runtime.graph_executor.GraphModule(lib["resnet-18"](dev))
module.set_input('data', data_np)

#module.set_input('fc_weight', weight_fc_np)
#module.set_input('fc_bias', bias_fc_np)



#print(module.benchmark(dev, number=100, repeat=3))

# Run inference
module.run()

# Get output
output = module.get_output(0)
#print(output)