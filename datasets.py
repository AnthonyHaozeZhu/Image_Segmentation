# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：datasets.py
@Author ：AnthonyZ
@Date ：2022/9/29 16:26
"""


from torch.utils.data import Dataset


# TODO: 完成数据集的加载任务
class Segmentation(Dataset):
    def __init__(self, args):
        self.opt = args
        data_path = args.data

    def __getitem__(self, index):
        return index