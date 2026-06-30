# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2023/7/31 23:02
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com

import numpy as np

from pyTensor.tensor import Tensor
from pyTensor.util import as_ndarray


class Function:
    def __call__(self, input):
        x = input.data
        y = as_ndarray(self.forward(x))  # scalar convert to ndarray.
        output = Tensor(y)
        self.input = input
        self.output = output
        output.set_creator(self)
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, dy):
        x = self.input.data
        dy = 2 * x * dy
        return dy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        x = self.input.data
        dy = np.exp(x) * dy
        return dy


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)



if __name__ == "__main__":
    x = Tensor(np.array(3.0))
    y = square(x)
    y.backward()
    print(x.grad)

