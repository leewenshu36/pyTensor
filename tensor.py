# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2023/7/31 23:02
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com

import numpy as np


class Tensor:
    def __init__(self, data):
        # 数据校验
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported!")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def zero_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            dys = [output.grad for output in f.outputs]
            dxs = f.backward(*dys)
            if not isinstance(dxs, tuple):
                dxs = (dxs,)

            for x, dx in zip(f.inputs, dxs):
                # 重复变量梯度覆盖
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad = x.grad + dx

                if x.creator is not None:
                    funcs.append(x.creator)


if __name__ == "__main__":
    list1 = ["a", "b", "c"]
    list2 = (1,)

    for i, j in zip(list1, list2):
        print(i)
        print(j)
