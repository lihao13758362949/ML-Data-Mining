# 数据预处理

from sklearn import preprocessing
#from sklearn.preprocessing import *
# 1. 删除重复数据
.drop_duplicates(subset=,keep=) # 删除重复数据
# subset：指出需要删除重复数据的列。
# keep：保留重复的哪一个数据，last,first,False，false表示删除所有重复数据。


# 2. 缺失值处理
# 删除变量
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
# 删除样本
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
.dropna()
train_df=train_df[~train_df['order_detail_status'].isin([101])]# 删除101订单状态订单

# 填充
all_data = all_data.fillna(all_data.mean())
# 根据业务实际情况填充。
#统计量：众数、中位数、均值
#插值法填充：包括随机插值，多重差补法，热平台插补，拉格朗日插值，牛顿插值等
#模型填充：使用回归、贝叶斯、随机森林、决策树等模型对缺失数据进行预测。
#定值填充
#np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
#int( 中位数/0.5 + 0.5 ) * 0.5

#还可以分组填充

#保留缺失值，用'None'填充

#偏正态分布，使用均值代替，可以保持数据的均值；偏长尾分布，使用中值代替，避免受 outlier 的影响

# 3.异常值和歧义值处理
# 标准化
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
# 删除
talExposureLog = totalExposureLog.loc[(totalExposureLog.pctr<=1000)]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

# 文本数据的清洗
	#在比赛当中，如果数据包含文本，往往需要进行大量的数据清洗工作。如去除HTML 标签，分词，拼写纠正, 同义词替换，去除停词，抽词干，数字和单位格式统一等
  
# 数据标准化
#标准正态分布标准化
Scaler=StandardScaler(copy=True, with_mean=True, with_std=True)

X_scaled = preprocessing.scale(X)  # Scaler=scale(X, axis=0, with_mean=True, with_std=True, copy=True)
#Min-max归一化（以下是0-1归一化）
Scaler=MinMaxScaler(feature_range=(0, 1), copy=True)
Scaler=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
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




# 4.编码

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} # 手动编码
dataset['Title'].map(title_mapping)
col_dicts = {}
cols = ['checking_balance','credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status', 
        'other_debtors','property','installment_plan','housing','job','telephone','foreign_worker']
col_dicts = {'checking_balance': {'1 - 200 DM': 2,
  '< 0 DM': 1,
  '> 200 DM': 3,
  'unknown': 0},
 'credit_history': {'critical': 0,
  'delayed': 2,
  'fully repaid': 3,
  'fully repaid this bank': 4,
  'repaid': 1}}
for col in cols:
    credit[col] = credit[col].map(col_dicts[col])
    
# credit.head(9)



OneHotEncoder(n_values=None, categorical_features=None, categories=None, drop=None, sparse=True, dtype=<class ‘numpy.float64’>, handle_unknown=’error’)# 热编码，若有n个类，则生成n个特征，其中一个是1其余是0.
# `sparse`：默认为True表示用稀疏矩阵表示，一般使用`.toarray()`转换到False，即数组。
OrdinalEncoder(categories=’auto’, dtype=<class ‘numpy.float64’>)# 序数编码
LabelEncoder().fit_transform(data[feature].astype(np.str)
#pd.get_dummies
#对于频数较少的那些分类变量可以归类到‘其他’pandas.DataFrame.replace后再进行编码。
#对于字符型的特征，要在编码后转换数据类型pandas.DataFrame.astype

# 连续变量离散化（分箱）
连续特征离散化pandas.cut，然后通过pandas.Series.value_counts观察切分点，再将对应bins或者说区间的连续值通过pandas.DataFrame.replace或pandas.DataFrame.loc到离散点0,1,2,…后再进行编码。

# 5. 不平衡数据处理
# 过采样
from imblearn.over_sampling import RandomOverSampler
X = df.iloc[:,:-1].values
y = df['quality'].values
ros = RandomOverSampler()
X, y = ros.fit_sample(X, y)
