import sys, subprocess
print('Python 版本:', sys.version)

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 Boston Housing 数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 先预览一下原始数据结构，确保我们理解每一列的含义和排布方式
print("原始数据前6行预览：")
print(raw_df.head(6))
print("\n原始数据形状:", raw_df.shape)

# Boston Housing 原始数据的每2行合成一个样本（sklearn 数据说明）
# 前12列特征在偶数行（0,2,4...），剩下2列加目标在奇数行（1,3,5...）
# 用 hstack 拼接成shape=(506,13)的特征数组
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# 目标变量在奇数行的第3列，提取为target（长度506）
target = raw_df.values[1::2, 2]

