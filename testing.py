import cupy as cp
from cupyx.scipy.special import expit
import cv2
import torch as t
from torch.nn import MaxPool2d, init as t_init
import os


class ForwardPropagation:
    def __init__(self, input_: cp.ndarray):
        """
        Initialize the ForwardPropagation class.

        Args:
            input_ (cp.ndarray): Input image or data as a CuPy array.
        """
        self.input = input_
        self.output = None
        self.weights_ = [self.weights(i) for i in range(4)]
        self.hiddenlayers = []

    @staticmethod
    def weights(index: int) -> cp.ndarray:
        """
        Load or initialize weights for a given layer index.
        If weights exist on disk, load them; otherwise, initialize and save.

        Args:
            index (int): The index of the layer.

        Returns:
            cp.ndarray: The weights for the specified layer.
        """
        layer = []
        if index == 0:
            if os.path.exists("weight0.npy"):
                return cp.load("weight0.npy")
            layer = cp.asarray([t_init.kaiming_uniform_(t.empty(6, 6, 1), mode='fan_in',
                                nonlinearity='linear').numpy().astype(cp.float16)
                                for _ in range(4096)])
            cp.save("weight0.npy", layer)

        elif index == 1:
            if os.path.exists("weight1.npy"):
                return cp.load("weight1.npy")
            layer = cp.asarray([t_init.kaiming_uniform_(t.empty(4096, 1), mode='fan_in',
                                nonlinearity='linear').numpy().reshape(4096, 1).astype(cp.float16)
                                for _ in range(4096)])
            cp.save("weight1.npy", layer)

        elif index == 2:
            if os.path.exists("weight2.npy"):
                return cp.load("weight2.npy")
            layer = cp.asarray([t_init.kaiming_uniform_(t.empty(4096, 1), mode='fan_in',
                                nonlinearity='linear').numpy().reshape(4096, 1).astype(cp.float16) for _ in range(1000)])
            cp.save("weight2.npy", layer)

        elif index == 3:
            if os.path.exists("weight4.npy"):
                return cp.load("weight4.npy")
            layer = cp.asarray([t_init.kaiming_uniform_(t.empty(1000, 1), mode='fan_in',
                                nonlinearity='linear').numpy().astype(cp.float16) for _ in range(2)])
            cp.save("weight4.npy", layer)

        return layer

    @staticmethod
    def convolve(image: cp.ndarray, kernel_size: int, pad: int = 0, stride: int = 0) -> cp.ndarray:
        """
        Apply Gaussian blur and unsharp masking to the input image.

        Args:
            image (cp.ndarray): Input image as a CuPy array.
            kernel_size (int): Size of the Gaussian kernel.
            pad (int, optional): Padding size. Defaults to 0.
            stride (int, optional): Stride for downsampling. Defaults to 0.

        Returns:
            cp.ndarray: The processed image as a CuPy array.
        """
        # Pad the image
        if pad != 0:
            padded = cp.pad(
                image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            padded = cp.asnumpy(padded).astype(cp.float32)
        else:
            padded = cp.asnumpy(image).astype(cp.float32)
        # Gaussian blur
        blurred = cp.asarray(cv2.GaussianBlur(
            padded, (kernel_size, kernel_size), 0))
        # Crop to original size
        if pad != 0:
            blurred = blurred[pad:-pad, pad:-pad, :]
        # Unsharp mask
        sharpened = cp.clip(image + 1.5 * (image - blurred), 0, 255)
        # print(sharpened.shape)
        if stride > 1:
            sharpened = sharpened[::stride, ::stride, :]
        return sharpened.astype(cp.uint8)

    """
    this might need to be changed later to use cupy for the entire process for less overhead
    """

    @staticmethod
    def pooling(image: cp.ndarray, pool_size: int, stride: int) -> cp.ndarray:
        """
        Apply max pooling to the input image using PyTorch.

        Args:
            image (cp.ndarray): Input image as a CuPy array.
            pool_size (int): Size of the pooling window.
            stride (int): Stride of the pooling operation.

        Returns:
            cp.ndarray: The pooled image as a CuPy array.
        """
        # Convert to NumPy, then to PyTorch tensor
        img_np = cp.asnumpy(image).astype(cp.uint8)
        img_torch = t.from_numpy(img_np).permute(
            2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
        pool = MaxPool2d(kernel_size=pool_size, stride=stride)
        pooled = pool(img_torch)
        # Remove batch dimension and permute back
        pooled_np = pooled.squeeze(0).permute(1, 2, 0).numpy()
        return cp.asarray(pooled_np)

    @staticmethod
    def perceptron(input_: cp.ndarray, weights_: cp.ndarray, bias: float) -> cp.ndarray:
        """
        Apply a perceptron (single-layer neural network) to the input.

        Args:
            input_ (cp.ndarray): Input data as a CuPy array.
            weights_ (cp.ndarray): Weights for the perceptron.
            bias (float): Bias term.

        Returns:
            cp.ndarray: Output of the perceptron as a CuPy array.
        """
        arr = []
        for weight in weights_:
            arr.append(expit(cp.sum(input_ * weight) + bias))
        return cp.asarray(arr).reshape(len(arr), 1).astype(cp.float16)

    def run(self) -> cp.ndarray:
        """
        Run the forward propagation process on the input.

        Args:
            input_ (cp.ndarray): Input image or data as a CuPy array.

        Returns:
            cp.ndarray: Final output after all layers.
        """
        con = self.convolve(self.input, 11, 0, 4)
        pooled = self.pooling(con, 3, 2)
        con = self.convolve(pooled, 5, 2)
        pooled = self.pooling(con, 3, 2)
        con = self.convolve(pooled, 3, 1)
        con = self.convolve(con, 3, 1)
        con = self.convolve(con, 3, 1)
        output = self.pooling(con, 3, 2)

        for i in range(4):
            activation = self.perceptron(output, self.weights_[i], 0)
            self.hiddenlayers.append(activation)
        return activation
