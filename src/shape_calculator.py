import numpy as np


def calculate_output_shape(inp_shape, c_out, k, s, p, d=1):
    c_in, h_in, w_in = inp_shape
    print(f"Input shape {inp_shape}")

    if isinstance(s, int):
        s = (s, s)

    new_h = (h_in + 2 * p - d * (k - 1) - 1) / s[0] + 1
    new_w = (w_in + 2 * p - d * (k - 1) - 1) / s[1] + 1

    h_out = int(np.floor(new_h))
    w_out = int(np.floor(new_w))

    if h_out != new_h or w_out != new_w:
        print(f"Flooring")

    out_shape = (c_out, h_out, w_out)
    print(f"Output shape {out_shape}")
    return out_shape


# think about un-equal stride to make image closer to square
input_dims = (3, 720, 1280)  # C x H x W
out_dims_1 = calculate_output_shape(input_dims, c_out=3, k=3, s=2, p=1)
out_dims_2 = calculate_output_shape(out_dims_1, c_out=1, k=3, s=3, p=1)
#out_dims_3 = calculate_output_shape(out_dims_2, c_out=3, k=3, s=(2, 1), p=1)

print("********")
dims = (6, 720, 1280)  # C x H x W
dims = calculate_output_shape(dims, c_out=64, k=7, s=3, p=3)
dims = calculate_output_shape(dims, c_out=1, k=3, s=2, p=1)
dims = calculate_output_shape(dims, c_out=1, k=3, s=1, p=1)
dims = calculate_output_shape(dims, c_out=1, k=3, s=1, p=1)

print("********")
dims = (6, 720, 1280)  # C x H x W
dims = calculate_output_shape(dims, c_out=64, k=7, s=(2,3), p=3)
dims = calculate_output_shape(dims, c_out=1, k=3, s=2, p=1)
dims = calculate_output_shape(dims, c_out=1, k=3, s=1, p=1)
dims = calculate_output_shape(dims, c_out=1, k=3, s=1, p=1)