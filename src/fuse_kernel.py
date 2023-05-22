import tvm

from tvm import topi, relay, te

import logging 
logging.basicConfig(level=logging.DEBUG)

batch = 1
in_channel = 256
in_size = 32
num_filter = 256
kernel = 3
stride = 1
padding = "SAME"
dilation = 1

A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
B = te.placeholder((1, num_filter, 1), name="bias")

def computer_inline_example():

    with tvm.target.Target("llvm"):
        
        conv1 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias1 = topi.add(conv1, B)
        relu1 = topi.nn.relu(bias1)

        s1 = te.create_schedule(relu1.op)
        print("----------------ORIGINAL------------------")
        print(tvm.lower(s1, [A, B, W], simple_mode=True))

        s1[bias1].compute_inline()
        print("--------------COMPUTE LINE-------------------")
        print(tvm.lower(s1, [A, B, W], simple_mode=True))

def fuse_example_1():
    with tvm.target.Target("llvm"):
        conv1 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias1 = topi.add(conv1, B)
        relu1 = topi.nn.relu(bias1)

        # Creating schedule
        s1 = te.create_schedule(relu1.op)
        #print(tvm.lower(s1, [A, B, W], simple_mode=True))

        # fuse relu1 into bias1 (It has dependency)
        s1[bias1].compute_at(s1[relu1], relu1.op.axis[2])

        print("--------------relu1 into bias1-------------------")
        # print the code in TensorIR format
        print(tvm.lower(s1, [A, B, W], simple_mode=True))

def fuse_example_2():
    with tvm.target.Target("llvm"):
        conv1 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias1 = topi.add(conv1, B)
        conv2 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias2 = topi.add(conv2, B)
        
        # I combined the bias1 and bias2 sequentially
        s1 = te.create_schedule([bias1.op, bias2.op])

        # fuse bias1 into bias2 (It has no dependency) 
        # ERROR here, It can't combined bias1 and bias2
        # Observation bias1 and bias2 have the same shape.
        s1[bias2].compute_at(s1[bias1], bias1.op.axis[2])

        print(tvm.lower(s1, [A, B, W], simple_mode=True))


def fuse_example_3():
    with tvm.target.Target("llvm"):
        conv1 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias1 = topi.add(conv1, B)
        conv2 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias2 = topi.add(conv2, B)
        
        # I combined the bias1 and bias2 sequentially
        s1 = te.create_schedule([bias1.op, bias2.op])

        # fuse bias1 into bias2 (It has no dependency) 
        # It's not worked as well
        s1[conv2].compute_inline()

        print(tvm.lower(s1, [A, B, W], simple_mode=True))


def computer_at_example_2():

    with tvm.target.Target("llvm"):
        
        # conv
        conv1 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias1 = topi.add(conv1, B)
        relu1 = topi.nn.relu(bias1)
        
        s1 = te.create_schedule(relu1.op)
        print("-"*100)
        print(tvm.lower(s1, [A, B, W], simple_mode=True))

        # pooling
        conv2 = topi.nn.conv2d(A, W, stride, padding, dilation)
        bias2 = topi.add(conv2, B)
        pool2 = topi.nn.pool2d(bias2, (3,3), (2,2), (1,1), (0,0,0,0), 'max')

        print("-"*100)
        s2 = te.create_schedule(pool2.op)
        print(tvm.lower(s2, [A, B, W], simple_mode=True))


computer_inline_example()
fuse_example_1()

#conv2_result = relay.nn.conv2d(relu_result, kernel, 1, 0)

#ss = tvm.create_schedule(conv2_result.op)

#ss[output_data].compute_at(ss[relu_result], relu_result.op.axis[3])

#ss[relu_result].compute_at(ss[conv2_result], conv2_result.op.axis[0])


