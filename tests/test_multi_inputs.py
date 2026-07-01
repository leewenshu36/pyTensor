# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2023/7/31 23:02
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com


import unittest
import numpy as np

from pyTensor.tensor import Tensor
from pyTensor.function import add, square


class MultiInputTest(unittest.TestCase):
    def test_compute(self):
        x, y = Tensor(np.array(2)), Tensor(np.array(3.0))
        z = add(square(x), square(y))
        expected = 13
        self.assertEqual(expected, z.data)


    def test_backward(self):
        x, y = Tensor(np.array(2)), Tensor(np.array(3.0))
        z = add(square(x), square(y))
        z_expected = np.array(13.0)

        z.backward()
        x_expected = np.array(4.0)
        y_expected = np.array(6.0)

        self.assertEqual(z.data, z_expected)
        self.assertEqual(x_expected, x.grad)
        self.assertEqual(y_expected, y.grad)


if __name__ == "__main__":
    unittest.main()