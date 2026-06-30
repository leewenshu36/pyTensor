# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2023/7/31 23:02
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com

import numpy as np


def as_ndarray(x):
    if np.isscalar(x):
        return np.array(x)
    return x