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
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path


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


def show_editied_masked_image(
    title: str,
    source_image: Image.Image,
    edited_image: Image.Image,
    mask: Optional[Image.Image] = None,
    path: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    fig_idx = 1
    rows = 1
    cols = 3 if mask is not None else 2

    fig = plt.figure(figsize=(12, 5))
    figure_title = f'Prompt: "{title}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)

    if mask is not None:
        fig.add_subplot(rows, cols, fig_idx)
        _set_image_plot_name("Mask")
        plt.imshow(mask)
        plt.gray()
        fig_idx += 1

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show(block=True)

    plt.close()


def _set_image_plot_name(name):
    plt.title(name)
    plt.xticks([])
    plt.yticks([])
