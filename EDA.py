# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df=pd.read_excel(r'C:\Users\lihao\Desktop\学年论文\链家二手房房源信息精简.xlsx',header=0,names=['price','layout','floor','direction','fitup','area','type','region','look_7','look_30'])

# 初步探索
df.head()

# 多变量探索

