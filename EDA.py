# 数据探索性分析

# 1.导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 2.读取数据
df=pd.read_excel(r'C:\Users\lihao\Desktop\学年论文\链家二手房房源信息精简.xlsx',header=0,names=['price','layout','floor','direction','fitup','area','type','region','look_7','look_30'])
# 连接数据
df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])
# 多文件读取
def get_data(path, get_type=True):
    features = []
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        if get_type:
            features.append([df['x'].std(), df['x'].mean(),
                             df['y'].std(), df['y'].mean(),
                             df['速度'].mean(), df['速度'].std(), 
                             df['方向'].mean(), df['方向'].std(),
                             file,
                             df['type'][0]])
        else:
            features.append([df['x'].std(), df['x'].mean(),
                             df['y'].std(), df['y'].mean(),
                             df['速度'].mean(), df['速度'].std(), 
                             df['方向'].mean(), df['方向'].std(),
                             file])
    df = pd.DataFrame(features)
    if get_type:
        df = df.rename(columns={len(features[0])-1:'label'})
        df = df.rename(columns={len(features[0])-2:'filename'})
        label_dict = {'拖网':0, '刺网':1, '围网':2}
        df['label'] = df['label'].map(label_dict)
    else:
        df = df.rename(columns={len(features[0])-1:'filename'})
    

    return df
df_train = get_data(trn_path)
df_test = get_data(test_path, False)

# 3.数据集基本信息（当前因变量为SalePrice）
df.info()

columns = df.columns.values.tolist() #获取所有的变量名

df.shape #：shape

df.head() #：给前几个样本
df.tail() #：给后几个样本
df.sample(10) #：随机给几个样本

df.describe() #：连续变量的一些描述信息，如基本统计量、分布等。
df.describe(include=['O']) #：分类变量的一些描述信息。
df.describe(include='all') #：全部变量的一些描述信息。

df.SalePrice.value_counts() #：观察取值数量

# df.SalePrice.value_counts(1) #：观察取值比例

credit.isnull().sum()/float(len(credit)) # 缺失值

df_value_ravel = df.values.ravel() 
print (u'缺失值数量：',len(df_value_ravel[df_value_ravel==np.nan])) # 缺失值


# 作图
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
