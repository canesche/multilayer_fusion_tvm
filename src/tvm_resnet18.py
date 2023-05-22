import tvm
from tvm import relay, autotvm, te, topi
import tvm.relay.testing
from tvm.relay.testing import layers
from tvm.relay.testing.init import create_workload

import numpy as np

batch_size = 1
input_shape = (batch_size, 3, 224, 224)
output_shape = (batch_size, 1000)

def resnet_18_tvm(name, dtype, layout):

    n_layer = int(name.split("-")[1])
    mod, params = relay.testing.resnet.get_workload(
        num_layers=n_layer, batch_size=batch_size, dtype=dtype, layout=layout
    )
    return mod, params, input_shape, output_shape

if __name__ == "__main__":

    name = "resnet-18"
    dtype = "float32"
    layout = "NCHW"
    batch_size = 1

    target = "llvm"

    
    mod, params, input_shape, output_shape = resnet_18_tvm(name, dtype, layout)

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    for i, t in enumerate(tasks):
        print(i+1, t)