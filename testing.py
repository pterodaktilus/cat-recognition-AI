
import cupy as cp
from cupyx.scipy.special import expit
import numpy as np
import cv2
import torch
import os

def weights(index:int) -> cp.ndarray:
    if index == 0:
        if os.path.exists("weight0.npy"):
            return cp.load("weight0.npy")
        layer0 = [torch.nn.init.kaiming_uniform_(torch.empty(27*27,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for i in range(4096)]
        np.save("weight0.npy", np.asarray(layer0) )
        return cp.asarray(layer0)
    if index == 1:
        if os.path.exists("weight1.npy"):
            return cp.load("weight1.npy")
        layer1 = [torch.nn.init.kaiming_uniform_(torch.empty(4096,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for i in range(4096)]
        np.save("weight1.npy", np.asarray(layer1) )
        return cp.asarray(layer1)
    if index == 2:
        if os.path.exists("weight2.npy"):
            return cp.load("weight2.npy")
        layer2 = [torch.nn.init.kaiming_uniform_(torch.empty(4096,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for i in range(1000)]
        np.save("weight2.npy", np.asarray(layer2) )
        return cp.asarray(layer2)
    if index == 3:
        if os.path.exists("weight3.npy"):
            return cp.load("weight3.npy")
        layer3 = [torch.nn.init.kaiming_uniform_(torch.empty(4096,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for i in range(1000)]
        np.save("weight3.npy", np.asarray(layer3) )
        return cp.asarray(layer3)
    if index == 4:
        if os.path.exists("weight4.npy"):
            return cp.load("weight4.npy")
        final = [torch.nn.init.kaiming_uniform_(torch.empty(2,1), mode='fan_in', nonlinearity='linear').numpy().astype(np.float16) for i in range(2)]
        np.save("weight4.npy", np.asarray(final) )
        return cp.asarray(final)
    else:
        raise ValueError("Invalid index for weights. Must be between 0 and 4.")



def convolve(image: cp.ndarray, kernel_size: int,stride: int) -> cp.ndarray:
    # Pad the image
    pad = kernel_size // 2
    padded = cp.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded = cp.asnumpy(padded).astype(np.float32)
    # Gaussian blur
    blurred = cp.asarray(cv2.GaussianBlur(padded, (kernel_size, kernel_size), 0))
    # Crop to original size
    blurred = blurred[pad:-pad, pad:-pad]
    # Unsharp mask
    sharpened = cp.clip(image + 1.5 * (image - blurred), 0, 255)
    print(sharpened.shape)
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

def perceptron(input: cp.ndarray, weights: cp.ndarray, bias: float):
    return expit(cp.sum(input * weights) + bias)

def test(input: cp.ndarray) -> cp.ndarray:
    con = convolve(input,11,4)
    pooled = pooling(con, 3, 2)
    con = convolve(pooled, 5, 2)
    pooled = pooling(con, 3, 2)
    con = convolve(pooled, 3, 1)
    con = convolve(con, 3, 1)
    con = convolve(con, 3, 1)
    output = pooling(con, 3, 2)
    for i in range(5):
        weights_ = weights(i)
        output = perceptron(output, weights_.reshape((weights_.shape[0],weights_.shape[1],1)), 0)
    print(output)
    return output


