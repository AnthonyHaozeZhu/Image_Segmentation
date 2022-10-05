# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/9/29 16:18
"""

import argparse

import torchvision
from PIL import Image
from mask import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--data", type=str, default="", help="Path of the dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512]
    )
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )
    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Control the clip guidance rate",
        default=1000
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--iterations_num",
        type=int,
        help="The number of iterations",
        default=8)
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true"
    )  # 如果加了--ddim则是True，否则是False
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )

    args = parser.parse_args()

    test = MaskMaker(args)

    a = Image.open("img.png").convert("RGB")
    a = torchvision.transforms.functional.to_tensor(a).type(torch.float16)
    result = test.make_mask_by_language(a, "a Cheetah")

    # TODO: 完成整个的测试过程

