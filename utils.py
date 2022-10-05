# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/10/4 16:38
"""

import torch.nn.functional as F
from collections import defaultdict
import numpy as np


def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class MetricsAccumulator:
    def __init__(self) -> None:
        self.accumulator = defaultdict(lambda: [])

    def update_metric(self, metric_name, metric_value):
        self.accumulator[metric_name].append(metric_value)

    def print_average_metric(self):
        for k, v in self.accumulator.items():
            average_v = np.array(v).mean()
            print(f"{k} - {average_v:.2f}")

        self.__init__()