# 初始设置


#pandas设置
import pandas as pd

#pd.set_option("display.max_columns",32) # 显示最大列数
pd.options.display.max_rows = 1000 # 显示最大行数
pd.options.display.max_columns = 20 # 显示最大列数
pd.set_option('display.float_format', lambda x: '%.3f' % x) # 小数格式设置

# warning设置
import warnings
warnings.filterwarnings("ignore") # 过滤警告文字

# plt设置
# %matplotlib inline # 不用show就可以显示图片，这一条需要手动在notebook中输入

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False # 在图片中显示中文

plt.style.use('fivethirtyeight')
