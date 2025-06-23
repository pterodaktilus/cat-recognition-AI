import cupy as cp
import PIL.Image as Image
import os
from testing import test
from checking import backpropagate

from typing import Tuple


def loadimage(path: str) -> cp.ndarray:
    """
    Loads an image from the specified file path, resizes it to 227x227 pixels, and converts it into a CuPy ndarray.

    Args:
        path (str): The file path to the image.

    Returns:
        cp.ndarray: The image as a CuPy ndarray with shape (227, 227, 3).

    Raises:
        FileNotFoundError: If the specified image file does not exist.
        OSError: If the image cannot be opened and identified.
    """
    img = Image.open(path).resize((227, 227))
    size = img.size
    return cp.asarray(img.getdata()).reshape(size[0], size[1], 3)


def main():

    path = os.getcwd()+"\\PetImages\\Dog"

    img = loadimage(path+"\\0.jpg")

    result = test(img)

    check_result = backpropagate(result)


if __name__ == "__main__":
    main()
