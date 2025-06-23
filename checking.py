import cupy as cp
from cupyx.scipy.special import expit
from testing import ForwardPropagation


class Backpropagation:
    def __init__(self, forward: ForwardPropagation):
        self.forward = forward
        self.weights = forward.weights_
        self.hiddenlayers = forward.hiddenlayers

    @staticmethod
    def errorcheck(result: cp.ndarray, expected: cp.ndarray, tol=0.05):
        return 0.5 * (expected - result) ** 2 > cp.ndarray((tol, tol))

    @staticmethod
    def result_sig_err(result: cp.ndarray, expected: cp.ndarray):
        return (expected - result) * expected * (1 - expected)

    @staticmethod
    def hidden_sig_err(result_err_grad: cp.ndarray, weight: cp.ndarray, sigm_res: cp.ndarray):
        return result_err_grad * weight * (1 - sigm_res) * sigm_res

    @staticmethod
    def weight_grad(result: cp.ndarray, sigm_res: cp.ndarray):
        return result * sigm_res

    def backpropagate(self, result: cp.ndarray, expected: cp.ndarray):
        bias_res_grad = self.result_sig_err(result, expected)
        weight_res_grad = self.hidden_sig_err(bias_res_grad, self.weights, self.hiddenlayers)
        self.forward.weights = self.forward.weights + weight_res_grad
