# 数据探索性分析

# 1.导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

# 2.读取数据
import os
import pandas as pd

# train_path=''
# test_path=''
def get_data(path,data_type='csv',header=None):
    ## 注意path格式，可能要加r
    if data_type == 'csv':
        df = pd.read_csv(path,header=header)
    if data_type=='xlsx'
        df=pd.read_excel(path,header=header,names=['price','layout','floor','direction','fitup','area','type','region','look_7','look_30'])
    if data_type='arff'
        # 读取arff文件
        from scipy.io import arff
        data,meta=arff.loadarff(path)
        df=pd.DataFrame(data)

    return df
df_train = get_data(trn_path)
df_test = get_data(test_path)
# 连接数据
all_data = pd.concat([df_train.assign(is_train=1), df_test.assign(is_train=0)])
# 多文件读取
def get_data2(path, get_type=True):
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
## 3.4 重复值
idsUnique = len(set(train.Id))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

df.duplicated() 
## 3.5 缺失值
credit.isnull().sum()/float(len(credit))

df_value_ravel = df.values.ravel() 
print (u'缺失值数量：',len(df_value_ravel[df_value_ravel==np.nan]))

missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

df_train.isnull().sum().max()# final check

## 3.6 异常值outlier

# 箱图（分类变量）
var = 'region'
data = pd.concat([df['price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="price", data=data)
# fig.axis(ymin=0, ymax=800000);
# 4. columns处理
# 数字变量和字符变量分开处理
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

# 5. 分布和偏态情况
y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)# QQ图，查看分布是否一致的

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())




# 6. 相关变量探索
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot(X,y,X_cols,y_col,plot_type=scatter):
    #scatter:散点图，pairplot:sns.pairplot，box：箱图，hist：直方图，heatmap：相关分析，热度图，list:列表汇总
    if plot_type==scatter:
        # 散点图（数字变量）
        data = pd.concat([X, y], axis=1)
        data.plot.scatter(x=X_cols, y=y_col);
    if plot_type==pairplot:
        #sns.pairplot
        sns.set()
        data = pd.concat([X, y], axis=1)
        sns.pairplot(data,x_vars=X_cols,y_vars=y_col)#参数size=2.5表示大小，aspect=0.8表示，kind='reg'添加拟合直线和95%置信区间
        plt.show();
    if plot_type=box:
        # 箱图（分类变量）
        var = 'region'
        data = pd.concat([df[y_col], df[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y=y_col, data=data)
        # fig.axis(ymin=0, ymax=800000);
    if plot_type=hist:
        # 对比，直方图
        g = sns.FacetGrid(train_df, col='Survived')
        g.map(plt.hist, 'Age', bins=20)
    if plot_type=heatmap:
        # 相关分析，热度图heatmaps1
        corrmat = df_train.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        # 选出和目标变量最相关的k个变量
        k = 10 #number of variables for heatmap
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df_train[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
    if plot_type=list:
        # 列表汇总（分类变量+数字变量）
        train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)



