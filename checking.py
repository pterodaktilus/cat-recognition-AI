import cupy as cp
from cupyx.scipy.special import expit



def errorcheck(result:cp.ndarray, expected: cp.ndarray, tol = 0.05):
    return 0.5 * (expected - result) ** 2 > cp.ndarray((tol,tol))
    
def weight_error(weight:cp.ndarray, result:cp.ndarray, learning_rate: float):
    pass

def backpropagate():
    pass