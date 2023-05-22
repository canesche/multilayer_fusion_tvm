import tvm
from tvm import te
from tvm import relay
import numpy as np

# Define the ResNet-18 architecture
def resnet_block(inputs, channels, strides=(1, 1), first_layer=False):
    # Convolutional layer 1
    conv1 = relay.nn.conv2d(
        inputs,
        relay.var("weight"),
        strides=strides,
        padding=(1, 1),
        channels=channels,
        kernel_size=(3, 3),
        data_layout="NHWC",
    )
    bn1 = relay.nn.batch_norm(conv1, relay.var("gamma"), relay.var("beta"), relay.var("mean"), relay.var("variance"))
    relu1 = relay.nn.relu(bn1)
    
    # Convolutional layer 2
    conv2 = relay.nn.conv2d(
        relu1,
        relay.var("weight"),
        strides=(1, 1),
        padding=(1, 1),
        channels=channels,
        kernel_size=(3, 3),
        data_layout="NHWC",
    )
    bn2 = relay.nn.batch_norm(conv2, relay.var("gamma"), relay.var("beta"), relay.var("mean"), relay.var("variance"))
    
    # Shortcut connection
    if first_layer:
        shortcut = relay.nn.conv2d(
            relu1,
            relay.var("weight"),
            strides=strides,
            padding=(1, 1),
            channels=channels,
            kernel_size=(1, 1),
            data_layout="NHWC",
        )
        shortcut_bn = relay.nn.batch_norm(
            shortcut, relay.var("gamma"), relay.var("beta"), relay.var("mean"), relay.var("variance")
        )
    else:
        shortcut_bn = inputs
    
    # Residual connection
    add = relay.add(bn2, shortcut_bn)
    relu2 = relay.nn.relu(add)
    
    return relu2


def resnet_18():
    data = relay.var("data", shape=(1, 224, 224, 3), dtype="float32")
    conv = relay.nn.conv2d(
        data,
        relay.var("weight"),
        strides=(2, 2),
        padding=(3, 3),
        channels=64,
        kernel_size=(7, 7),
        data_layout="NHWC",
    )
    bn = relay.nn.batch_norm(conv, relay.var("gamma"), relay.var("beta"), relay.var("mean"), relay.var("variance"))
    relu = relay.nn.relu(bn)
    pool = relay.nn.max_pool2d(relu, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), layout="NHWC")
    
    channels = [64, 64, 128, 256, 512]
    strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
    for i in range(4):
        first_layer = True if i == 0 else False
        pool = resnet_block(pool, channels[i + 1], strides[i], first_layer)
    
    # Global average pooling
    pool = relay.nn.avg_pool2d(pool, pool_size=(7, 7), strides=(1, 1), layout="NHWC")
    flatten = relay.nn.batch_flatten(pool)
    dense = relay.nn.dense(flatten, relay.var("weight"), units=1000)
    softmax = relay.nn.softmax(dense)
    
    return relay.Function(relay.analysis.free_vars(softmax), softmax)


# Compile the model with TVM
target = "llvm"
target_host = "llvm"


resnet = resnet_18()
mod, params = relay.frontend.from_expr(resnet)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

# Create TVM runtime and module
ctx = tvm.cpu()
module = tvm.runtime.GraphModule(lib["default"](ctx))

# Set the input shape
input_shape = (1, 224, 224, 3)

# Create a random input tensor
input_data = tvm.nd.array(np.random.uniform(size=input_shape).astype("float32"))

# Set the input data
module.set_input("data", input_data)

# Run inference
module.run()

# Get the output
output = module.get_output(0)

print("Output shape:", output.shape)
print("Output values:", output.asnumpy())