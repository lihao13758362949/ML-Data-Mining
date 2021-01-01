# 初始设置

import pandas as pd

pd.set_option("display.max_columns",32) # 显示最大列数

import warnings
warnings.filterwarnings("ignore") # 过滤警告文字

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False # 在图片中显示中文
