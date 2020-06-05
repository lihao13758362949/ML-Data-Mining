# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 读取数据
df=pd.read_excel(r'C:\Users\lihao\Desktop\学年论文\链家二手房房源信息精简.xlsx',header=0,names=['price','layout','floor','direction','fitup','area','type','region','look_7','look_30'])

# 数据预预处理
df.head()

df.head()
# 初步探索（当前因变量为SalePrice）
df.info()
df.columns#：显示所有的变量名
df.shape#：shape
df.head()#：给前几个样本
df.tail()#：给后几个样本
df.sample(10)#：随机给几个样本
df.describe()#：连续变量的一些描述信息，如基本统计量、分布等。
df.describe(include=['O'])#：分类变量的一些描述信息。
df.describe(include='all')#：全部变量的一些描述信息。
Y_train.value_counts()#：观察取值数量

df_train['SalePrice'].describe()

sns.distplot(df['price'])
# 多变量探索
# 散点图（数字变量）
data = pd.concat([df['price'], df['area']], axis=1)
data.plot.scatter(x='area', y='price', ylim=(0,40000));

# 箱图（分类变量）
var = 'region'
data = pd.concat([df['price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="price", data=data)
# fig.axis(ymin=0, ymax=800000);

# 对比，直方图
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# 列表汇总（分类变量+数字变量）
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 相关分析，热度图heatmaps1
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
# 选出和目标变量最相关的k个变量
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
