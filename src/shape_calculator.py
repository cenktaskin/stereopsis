import numpy as np
input_dims = (3, 360, 640)  # C x H x W
c_in = input_dims[0]
h_in = input_dims[1]
w_in = input_dims[2]

filter_count = 3
kernel_size = 3
stride = 3
padding = 1
dilation = 1

c_out = filter_count
h_out = int(np.floor((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
w_out = int(np.floor((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
h_out0 = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
w_out0 = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

if h_out != h_out0 or w_out != w_out0:
    print(f"Flooring")

print(f"Input shape {input_dims}")
print(f"Output shape {(c_out,h_out, w_out)}")