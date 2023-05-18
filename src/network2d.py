import numpy as np
import time

np.random.seed(0)

# Define the input and output shapes
batch_size = 1
channels = 3
height = 224
width = 224
num_classes = 10
input_shape = (batch_size, channels, height, width)
output_shape = (batch_size, num_classes)

# Define the parameters
weight1 = np.random.randn(64, channels, 3, 3).astype('float32')
bias1 = np.zeros((64,)).astype('float32')
weight2 = np.random.randn(num_classes, 64, 7, 7).astype('float32')
bias2 = np.zeros((num_classes,)).astype('float32')

# Define the computation graph
def conv2d(input_data, weight, bias, stride, padding):
    batch_size, in_channels, in_height, in_width = input_data.shape
    out_channels, _, filter_height, filter_width = weight.shape
    out_height = (in_height - filter_height + 2 * padding) // stride + 1
    out_width = (in_width - filter_width + 2 * padding) // stride + 1

    output_data = np.zeros((batch_size, out_channels, out_height, out_width)).astype('float32')
    for b in range(batch_size):
        for c_out in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride - padding
                    h_end = h_start + filter_height
                    w_start = w_out * stride - padding
                    w_end = w_start + filter_width

                    conv_sum = 0.0
                    for c_in in range(in_channels):
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                if h >= 0 and h < in_height and w >= 0 and w < in_width:
                                    conv_sum += input_data[b, c_in, h, w] * weight[c_out, c_in, h-h_start, w-w_start]

                    output_data[b, c_out, h_out, w_out] = conv_sum + bias[c_out]
    return output_data

def conv2d_opt(input_data, weight, bias, stride, padding):
    batch_size, in_channels, in_height, in_width = input_data.shape
    out_channels, _, filter_height, filter_width = weight.shape
    out_height = (in_height - filter_height + 2 * padding) // stride + 1
    out_width = (in_width - filter_width + 2 * padding) // stride + 1

    output_data = np.zeros((batch_size, out_channels, out_height, out_width)).astype('float32')

    # Tiling parameters
    tile_size_batch = 4
    tile_size_out_channels = 4
    tile_size_in_channels = 4
    tile_size_out_width = 16

    # Loop over tiles
    for b_start in range(0, batch_size, tile_size_batch):
        b_end = min(b_start + tile_size_batch, batch_size)
        for c_out_start in range(0, out_channels, tile_size_out_channels):
            c_out_end = min(c_out_start + tile_size_out_channels, out_channels)
            for c_in_start in range(0, in_channels, tile_size_in_channels):
                c_in_end = min(c_in_start + tile_size_in_channels, in_channels)
                for w_out_start in range(0, out_width, tile_size_out_width):
                    w_out_end = min(w_out_start + tile_size_out_width, out_width)

                    # Compute the tile
                    for b in range(b_start, b_end):
                        for c_out in range(c_out_start, c_out_end):
                            for h_out in range(out_height):
                                for w_out in range(w_out_start, w_out_end):
                                    h_start = h_out * stride - padding
                                    h_end = h_start + filter_height
                                    w_start = w_out * stride - padding
                                    w_end = w_start + filter_width

                                    conv_sum = 0.0
                                    for c_in in range(c_in_start, c_in_end):
                                        for h in range(h_start, h_end):
                                            for w in range(w_start, w_end):
                                                if h >= 0 and h < in_height and w >= 0 and w < in_width:
                                                    conv_sum += input_data[b, c_in, h, w] * weight[c_out, c_in, h-h_start, w-w_start]

                                    output_data[b, c_out, h_out, w_out] += conv_sum

    # Add bias
    for b in range(batch_size):
        for c_out in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    output_data[b, c_out, h_out, w_out] += bias[c_out]

    return output_data

def relu(input_data):
    return np.maximum(input_data, 0)

def max_pool2d(input_data, pool_size, stride):
    batch_size, channels, in_height, in_width = input_data.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    output_data = np.zeros((batch_size, channels, out_height, out_width)).astype('float32')
    for b in range(batch_size):
        for c in range(channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride
                    h_end = h_start + pool_size
                    w_start = w_out * stride
                    w_end = w_start + pool_size

                    output_data[b, c, h_out, w_out] = np.max(input_data[b, c, h_start:h_end, w_start:w_end])

    return output_data


def max_pool2d_opt2(input_data, pool_size, stride):
    batch_size, channels, in_height, in_width = input_data.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    output_data = np.zeros((batch_size, channels, out_height, out_width)).astype('float32')
    
    # Define tile size
    tile_h = 32
    tile_w = 32
    
    # Perform tiling and parallelize computation over tiles
    for b in range(batch_size):
        for c in range(channels):
            for h_tile in range(0, out_height, tile_h):
                for w_tile in range(0, out_width, tile_w):
                    h_start = h_tile * stride
                    h_end = min(h_start + pool_size, in_height)
                    w_start = w_tile * stride
                    w_end = min(w_start + pool_size, in_width)
                    
                    # Compute max pooling for each tile
                    output_tile = np.zeros((tile_h, tile_w)).astype('float32')
                    for h in range(h_start, h_end):
                        for w in range(w_start, w_end):
                            output_tile[h-h_start, w-w_start] = np.max(input_data[b, c, h, w])
                    
                    # Write the result of the tile to the output
                    h_tile_end = min(h_tile+tile_h, out_height)
                    w_tile_end = min(w_tile+tile_w, out_width)
                    output_data[b, c, h_tile:h_tile_end, w_tile:w_tile_end] = output_tile[:h_tile_end-h_tile, :w_tile_end-w_tile]
    
    return output_data

def max_pool2d_opt(input_data, pool_size, stride):
    batch_size, channels, in_height, in_width = input_data.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    output_data = np.zeros((batch_size, channels, out_height, out_width)).astype('float32')
    
    # Define tile sizes
    tile_b = 4
    tile_c = 32
    tile_h = 32
    tile_w = 32
    
    # Perform tiling and parallelize computation over tiles
    for b_tile in range(0, batch_size, tile_b):
        b_tile_end = min(b_tile+tile_b, batch_size)
        for c_tile in range(0, channels, tile_c):
            c_tile_end = min(c_tile+tile_c, channels)
            for h_tile in range(0, out_height, tile_h):
                h_tile_end = min(h_tile+tile_h, out_height)
                for w_tile in range(0, out_width, tile_w):
                    w_tile_end = min(w_tile+tile_w, out_width)
                    
                    # Compute max pooling for each tile
                    for b in range(b_tile, b_tile_end):
                        for c in range(c_tile, c_tile_end):
                            for h in range(h_tile, h_tile_end):
                                for w in range(w_tile, w_tile_end):
                                    h_start = h * stride
                                    h_end = min(h_start + pool_size, in_height)
                                    w_start = w * stride
                                    w_end = min(w_start + pool_size, in_width)
                                    output_data[b, c, h, w] = np.max(input_data[b, c, h_start:h_end, w_start:w_end])
    
    return output_data

def dense(input_data, weight, bias):
    batch_size, in_features = input_data.shape
    out_features, _ = weight.shape

    output_data = np.zeros((batch_size, out_features)).astype('float32')
    for b in range(batch_size):
        for f_out in range(out_features):
            dense_sum = 0.0
            for f_in in range(in_features):
                    if f_in < weight.shape[1]:
                        dense_sum += input_data[b, f_in] * weight[f_out, f_in]
            output_data[b, f_out] = dense_sum / in_features + bias[f_out]
    return output_data

def dense_opt(input_data, weight, bias):
    batch_size, in_features = input_data.shape
    out_features, _ = weight.shape

    output_data = np.zeros((batch_size, out_features)).astype('float32')
    
    # Define tile sizes
    tile_b = 32
    tile_f_out = 256
    tile_f_in = 256
    
    # Perform tiling and parallelize computation over tiles
    for b_tile in range(0, batch_size, tile_b):
        for f_out_tile in range(0, out_features, tile_f_out):
            for f_in_tile in range(0, in_features, tile_f_in):
                b_tile_end = min(b_tile + tile_b, batch_size)
                f_out_tile_end = min(f_out_tile + tile_f_out, out_features)
                f_in_tile_end = min(f_in_tile + tile_f_in, in_features)

                # Compute dense layer for each tile
                for b in range(b_tile, b_tile_end):
                    for f_out in range(f_out_tile, f_out_tile_end):
                        dense_sum = 0.0
                        for f_in in range(f_in_tile, f_in_tile_end):
                            if f_in < weight.shape[1]:
                                dense_sum += input_data[b, f_in] * weight[f_out, f_in]
                        output_data[b, f_out] = dense_sum
                    
                # Add bias term to output
                for f_out in range(f_out_tile, f_out_tile_end):
                    output_data[b_tile:b_tile_end, f_out] = np.add(output_data[b_tile:b_tile_end, f_out], bias[f_out])
                    
                # Normalize output by dividing with in_features
                for b in range(b_tile, b_tile_end):
                    for f_out in range(f_out_tile, f_out_tile_end):
                        output_data[b, f_out] /= in_features
    
    return output_data

def softmax(input_data):
    exp_data = np.exp(input_data)
    return exp_data / np.sum(exp_data)

def two_layer_net(input_data, weight1, bias1, weight2, bias2):
    
    # layer-1
    conv1 = conv2d_opt(input_data, weight1, bias1, stride=1, padding=1)
    relu1 = relu(conv1)
    pool1 = max_pool2d_opt(relu1, pool_size=2, stride=2)

    # layer-2
    pool1_flattened = pool1.reshape((batch_size, -1))
    weight2_flattened = weight2.reshape((num_classes, -1))
    dense1 = dense_opt(pool1_flattened, weight2_flattened, bias2)
    output_data = softmax(dense1)
    return output_data

# Test the model with random input
input_data = np.random.randn(*input_shape).astype('float32')

start = time.time()
output_data = two_layer_net(input_data, weight1, bias1, weight2, bias2)
final = time.time() - start

print(final)

print(output_data)
print(sum(output_data[0]))