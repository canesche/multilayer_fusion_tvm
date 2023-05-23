import os

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime


target = tvm.target.cuda()

N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
data = relay.var('data', shape=(N, CI, H, W))
#data = te.placeholder((N, CI, H, W), name='data')
#kernel = relay.const(np.random.random((CO, CI, KH, KW)).astype("float32"))
kernel = relay.var('kernel', shape=(CO, CI, KH, KW))
#kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
dilation=(1,1)
dtype = "float32"
kernel_shape = (CO, CI, KH, KW)

ctx = tvm.gpu()

out = relay.op.nn.nn.conv2d(data, kernel, strides=strides, padding=padding, dilation=dilation, channels = CO, kernel_size = (KH, KW), data_layout='NCHW', out_dtype=dtype)

mod = tvm.IRModule.from_expr(out)

kernel_weights = tvm.nd.array(np.ones(kernel_shape, dtype=dtype), ctx)

dict_params = {'kernel': kernel_weights}
dict_params = dict()

task = autotvm.task.extract_from_program(mod["main"], target=target, params=dict_params, ops=(relay.op.get('nn.conv2d'),))
print(task[0].config_space)
print(task[1].config_space)
env = autotvm.task.topi_integration.TaskExtractEnv.get()
print(env.get_tasks())