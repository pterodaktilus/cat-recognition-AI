
import cupy as cp
from cupyx.scipy.special import expit
import numpy as np
import cv2
import torch
import os

def weights(index:int) -> cp.ndarray:
    layer = []
    if index == 0:
        if os.path.exists("weight0.npy"):
            return cp.load("weight0.npy")
        layer = cp.asarray([torch.nn.init.kaiming_uniform_(torch.empty(6,6,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for _ in range(4096)])
        cp.save("weight0.npy", cp.asarray(layer) )

    if index == 1:
        if os.path.exists("weight1.npy"):
            return cp.load("weight1.npy")
        layer = cp.asarray([torch.nn.init.kaiming_uniform_(torch.empty(4096,1), mode='fan_in', nonlinearity='linear').numpy().reshape(4096,1).astype(np.float16) for _ in range(4096)])
        cp.save("weight1.npy", cp.asarray(layer) )

    if index == 2:
        if os.path.exists("weight2.npy"):
            return cp.load("weight2.npy")
        layer = cp.asarray([torch.nn.init.kaiming_uniform_(torch.empty(4096,1), mode='fan_in', nonlinearity='linear').numpy().reshape(4096,1).astype(np.float16) for _ in range(1000)])
        cp.save("weight2.npy", cp.asarray(layer) )

    if index == 3:
        if os.path.exists("weight4.npy"):
            return cp.load("weight4.npy")
        layer = cp.asarray([torch.nn.init.kaiming_uniform_(torch.empty(1000,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for _ in range(2)])
        cp.save("weight4.npy", cp.asarray(layer) )

    return layer


def convolve(image: cp.ndarray, kernel_size: int,pad:int = 0 ,stride: int = 0) -> cp.ndarray:
    # Pad the image
    if pad != 0:
        padded = cp.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        padded = cp.asnumpy(padded).astype(np.float32)
    else:
        padded = cp.asnumpy(image).astype(np.float32)
    # Gaussian blur
    blurred = cp.asarray(cv2.GaussianBlur(padded, (kernel_size, kernel_size), 0))
    # Crop to original size
    if pad != 0:
        blurred = blurred[pad:-pad, pad:-pad, :]
    # Unsharp mask
    sharpened = cp.clip(image + 1.5 * (image - blurred), 0, 255)
    #print(sharpened.shape)
    if stride > 1:
        sharpened = sharpened[::stride, ::stride, :]
    return sharpened.astype(np.uint8)


"""
this might need to be changed later to use cupy for the entire process for less overhead
"""
def pooling(image: cp.ndarray, pool_size: int, stride: int) -> cp.ndarray:
    # Convert to NumPy, then to PyTorch tensor
    img_np = cp.asnumpy(image).astype(np.uint8)
    img_torch = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    pool = torch.nn.MaxPool2d(kernel_size=pool_size, stride=stride)
    pooled = pool(img_torch)
    # Remove batch dimension and permute back
    pooled_np = pooled.squeeze(0).permute(1, 2, 0).numpy()
    return cp.asarray(pooled_np)

def perceptron(input_: cp.ndarray, weights_: cp.ndarray, bias: float) -> cp.ndarray:
    arr = []
    for weight in weights_:

        arr.append(expit(cp.sum(input_ * weight) + bias))
    return cp.asarray(arr).reshape(len(arr),1).astype(np.float16)

def test(input_: cp.ndarray) -> cp.ndarray:
    con = convolve(input_,11,0,4)
    pooled = pooling(con, 3, 2)
    con = convolve(pooled, 5, 2)
    pooled = pooling(con, 3, 2)
    con = convolve(pooled, 3, 1)
    con = convolve(con, 3, 1)
    con = convolve(con, 3, 1)
    output = pooling(con, 3, 2)

    for i in range(4):
        weights_ = weights(i)
        output = perceptron(output, weights_, 0)
    print(output.astype(float))
    return output


