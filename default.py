# 初始设置

import pandas as pd

pd.set_option("display.max_columns",32) # 显示最大列数
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
import warnings
warnings.filterwarnings("ignore") # 过滤警告文字

# plt设置
%matplotlib inline # 不用show就可以显示图片

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False # 在图片中显示中文
