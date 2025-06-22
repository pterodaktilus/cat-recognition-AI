import cupy as cp
import PIL.Image as Image
import os
from testing import test
from checking import check


from typing import Tuple

def loadimage(path: str) -> cp.ndarray:
    img = Image.open(path).resize((227, 227))
    size = img.size
    return cp.asarray(img.getdata()).reshape(size[0], size[1], 3)



def main():
    path = os.getcwd()+"\\PetImages\\Dog"
    img = loadimage(path+"\\0.jpg")
    result = test(img)
    check_result = check(result)

    

if __name__ == "__main__":
    main()