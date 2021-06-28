# 初始设置


# 1 <pandas设置>
import pandas as pd

#pd.set_option("display.max_columns",32) # 显示最大列数
pd.options.display.max_rows = 1000 # 显示最大行数，若为None则为无穷
pd.options.display.max_columns = 20 # 显示最大列数，若为None则为无穷
pd.set_option('display.float_format', lambda x: '%.2f' % x) # 小数格式设置

# 2 <warning设置>
import warnings
warnings.filterwarnings("ignore") # 过滤警告文字

# 3 <plt设置>
# %matplotlib inline # 不用show就可以显示图片，这一条需要手动在notebook中输入

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False # 在图片中显示中文
plt.rcParams['figure.figsize'] = (10.0, 5.0)

plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-dark') 

# 4 <sns设置>
sns.set()
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

# 5 <time类型设置>
# %%time #显示程序块运行时间，这一条需要手动在notebook中输入
