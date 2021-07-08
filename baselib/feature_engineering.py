"""
<特征工程>
输入：all_data
输出：train_final.csv,test_final.csv或 train_for_tree.csv,test_for_tree.csv,train_for_lr.csv,...

>特征决定你的上限，模型只不过在无限逼近这个值罢了。
>有人总结 Kaggle 比赛是 “Feature 为主，调参和 Ensemble 为辅”
>之所以构造不同的数据集是因为，不同模型对数据集的要求不同

> 自动特征工程FeatureTools
"""

import pandas as pd
import gc

 
# 2 <加变量> <特征提取>（Feature Extraction）
## 2.1 <简化>
train["SimplOverallQual"] = train.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
## 2.2 <组合>（特征间的加减乘除）
### 2.2.1 <连续变量的组合>
train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
### 2.2.2 <离散变量的组合>

## 2.3 <多项式>
train["OverallQual-s2"] = train["OverallQual"] ** 2
train["OverallQual-s3"] = train["OverallQual"] ** 3
train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
## 2.4 <正则提取>
# >正则提取
# >str.split分割字符，再用pandas.DataFrame.add_prefix添加前缀成为新变量

## 2.5 <其它>
# >如可以把一个变量是否为0作为特征
def geo_feature():
    # 地理位置相关特征
    # 按经纬度进行网格划分，用word2vec及逆行训练得到编码向量
    # 然后区域统计特征
    pass

def level1_feature():
    # 一阶统计特征
    # count，max，min，sum，mean，极差，偏度，峰度
    pass


def level2_feature():
    # 二阶统计特征（时间序列常用）
    # 差分的均值、方差
    pass


# all_data.groupby.agg()
data['count'] = 1
tmp = data[data['goods_has_discount']==1].groupby(['customer_id'])['count'].agg({'goods_has_discount_counts':'count'}).reset_index()
customer_all = customer_all.merge(tmp,on=['customer_id'],how='left')

for col in ['aid','goods_id','account_id']:
    result = logs.groupby([col,'day'], as_index=False)['isExp'].agg({
        col+'_cnts'      : 'count',
        col+'_sums'      : 'sum',
        col+'_rate'      : 'mean'
        })
    result[col+'_negs'] = result[col+'_cnts'] - result[col+'_sums']
    data = data.merge(result, how='left', on=[col,'day'])
    
# 计算某品牌的销售统计量，同学们还可以计算其他特征的统计量
# 这里要以 train 的数据计算统计量
train_gb = train.groupby("brand")
all_info = {}
for kind, kind_data in train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
data = data.merge(brand_fe, how='left', on='brand')

# >all_data.groupby.last/min/max....
def create_time_feat(df, time_feat):

    # 时间特征构造
    # 可以利用时间特征来划分数据集，滑动窗口
    # train_history = train[(train['order_pay_date'].astype(str)<='2013-07-03')]
    # train_label = train[train['order_pay_date'].astype(str)>='2013-07-04']
    df[time_feat] = pd.to_datetime(df[time_feat])
    df['date'] = df[time_feat].dt.date
    df['day'] = df[time_feat].dt.day
    df['hour'] = df[time_feat].dt.hour


def add_prefix(df, exclude_columns, prefix):
    # 给特征增加前缀
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]

    column_names = [col for col in df.columns if col not in exclude_columns]
    df.rename(columns=dict(zip(column_names, [prefix + name for name in column_names])), inplace=True)
    return df
# 2 <减变量><特征选择>（Feature Selection）
## 2.1 <降维算法> 

## 2.2 <直接删除>
# >利用好了，就可以删掉原始数据了
data = data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)
## 2.3 <过滤法>（filter）
# >先选择后训练。按照评估准则对各个特征进行评分，然后按照筛选准则来选择特征。
## 2.4 <包裹法>（wrapper）
# >一训练一筛选。根据学习器预测效果评分，每次选择若干特征，或者排除若干特征。一般性能比过滤法好，但计算开销较大。可以分为前向搜索、后向搜索及双向搜索。
## 2.5 <嵌入法>（embedded）
# >先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小排序选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。常见的嵌入式方法有L1正则化。
# >另外，该方法常用于处理稀疏表示和字典学习以及压缩感知。
# >Lasso 回归和决策树可以完成嵌入式特征选择
# >大部分情况下都是用嵌入式做特征筛选

# 4 <数据标准化/归一化>
'''
数据标准化/归一化的作用：
1. 在梯度下降中不同特征的更新速度变得一致，更容易找到最优解


应用场景：
通过梯度下降法求解的模型需要归一化：线性回归、逻辑回归、SVM、NN等

决策树不需要归一化，因为是否归一化不会改变信息增益
'''
# 4.1 <标准正态分布标准化>Z-Score Normalization
Scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
df[column]=StandardScaler().fit_transform(df[column][:,np.newaxis])

X_scaled = preprocessing.scale(X)  # Scaler=scale(X, axis=0, with_mean=True, with_std=True, copy=True)
'''
z=(x-miu)/sigma
'''
## 4.2 <Min-max归一化（0-1）>Min-Max Scaling
Scaler=MinMaxScaler(feature_range=(0, 1), copy=True)
Scaler=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
'''
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
'''
Scaler=MaxAbsScaler(copy=True)# [-1,1]
Scaler=maxabs_scale(X, axis=0, copy=True)
Scaler.fit(X_train)
Scaler.transform(X_test)
Scaler.fit_transform(X_train)

#standardizing data实例
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# 5.<特征编码>
'''
对于频数较少的那些分类变量可以归类到‘其他’pandas.DataFrame.replace后再进行编码。
对于字符型的特征，要在编码后转换数据类型pandas.DataFrame.astype
'''
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

def encode_by_hand(df, cols, map):
    # 手动编码
    # cols是要替换的列的list

    # maps = {'checking_balance': {'1 - 200 DM': 2,
    #   '< 0 DM': 1,
    #   '> 200 DM': 3,
    #   'unknown': 0},
    #  'credit_history': {'critical': 0,
    #   'delayed': 2,
    #   'fully repaid': 3,
    #   'fully repaid this bank': 4,
    #   'repaid': 1}}
    for col in cols:
        df[col] = df[col].map(map[col])
    return df

## 5.2 <序号编码>Ordinary Encoding
OrdinalEncoder(categories=’auto’, dtype=<class ‘numpy.float64’>)
## 5.3 <独热编码>One-hot Encoding
'''
会导致高维度特征，应配合特征选择来降低维度
'''
OneHotEncoder(n_values=None, categorical_features=None, categories=None, drop=None, sparse=True, dtype=<class ‘numpy.float64’>, handle_unknown=’error’)# 热编码，若有n个类，则生成n个特征，其中一个是1其余是0.
# `sparse`：默认为True表示用稀疏矩阵表示，一般使用`.toarray()`转换到False，即数组。
## 5.4 <二进制编码>Binary Encoding
'''
用二进制来表示不同的类别如3表示为011
维度少于独热编码
'''
## 5.6 其它编码方式，比如Helmert Contrast、Sum Contrast、Polynomial Contrast、Backward Difference Contrast等。


LabelEncoder().fit_transform(data[feature].astype(np.str)
# 对类别特征进行 OneEncoder
data = pd.get_dummies(data, columns=['model', 'brand', 'bodyType', 'fuelType',
                                     'gearbox', 'notRepairedDamage', 'power_bin'])


## 5.6 <连续变量离散化>（分箱）
# 等频分桶；
# 等距分桶；
# Best-KS 分桶（类似利用基尼指数进行二分类）；
# 卡方分桶；
连续特征离散化pandas.cut，然后通过pandas.Series.value_counts观察切分点，再将对应bins或者说区间的连续值通过pandas.DataFrame.replace或pandas.DataFrame.loc到离散点0,1,2,…后再进行编码。


def data_output(label, all_data):
    # 最终输出>
    print(all_data.shape)
    print(f'features: {list(all_data.columns)}')
    all_data[all_data['is_train']==1].drop('is_train', axis=1).to_csv('train_final.csv', index=0)
    all_data[all_data['is_train']==0].drop([label,'is_train'], axis=1).to_csv('test_final.csv', index=0)
                             
                             
                             
                             
                             
                             
                             
