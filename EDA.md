***
# 0 写在前面
参考资料：
1. [pandas分批读取大数据集](https://blog.csdn.net/htbeker/article/details/86542412)
2. [智慧海洋建设baseline——wbbhcb](https://github.com/wbbhcb/zhhyjs_baseline)
3. [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# 1 EDA概述
EDA (Exploratory Data Analysis)，也就是对数据进行探索性的分析，从而为之后的[数据预处理](https://blog.csdn.net/weixin_42297855/article/details/97629534)和[特征工程](https://blog.csdn.net/weixin_42297855/article/details/97505444)提供必要的结论。
主要的步骤是：
1. 理解问题；
2. 读取数据；
3. 单变量探索；
4. 多变量探索；
5. 数据预处理；
6. 建立假设，并检验。

拿到数据之后，我们必须要明确以下几件事情：

1. 数据是如何产生的，数据又是如何存储的；
2. 数据是原始数据，还是经过人工处理(二次加工的)；
3. 数据由那些业务背景组成的，数据字段又有什么含义；
4. 数据字段是什么类型的，每个字段的分布是怎样的；
5. 训练集和测试集的数据分布是否有差异;

在分析数据的过程中，还必须要弄清楚的以下数据相关的问题：

1. 数据量是否充分，是否有外部数据可以进行补充；
2. 数据本身是否有噪音，是否需要进行数据清洗和降维操作；
3. 赛题的评价函数是什么，和数据字段有什么关系；
4. 数据字段与赛题标签的关系；
# 2 读取数据
>核心方法是使用`pandas.read_csv`和`pandas.read_table`等方法读取数据。

>有多个文件组成数据则将他们连接，也经常把训练集和测试集合并起来处理，可用`pandas.concat`等方法，如`df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])`。


## 2.1 大数据读取
1. 分批读取
例如可以用到`read_csv`里的`chunksize`参数读取部分数据用于train，如
```py
chunks = pd.read_csv('train.csv',iterator = True)
chunk = chunks.get_chunk(5)
```
也可以合并各个chunk，如
```py
def get_df(file):
	mylist=[]
	for chunk in pd.read_csv(file,chunksize=1000000):
		mylist.append(chunk)
	temp_df=pd.concat(mylist,axis=0)
	del mylist
	return temp_df
```
2. h5
[df to hdf](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200107123430241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
## 2.2 编码
注意下pandas读取数据时的编码方法要和原始数据对应。

## 2.3 多文件读取
```py
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
```
#  3 数据探索
## 3.1 初步探索
完成以下几点任务：
1. 记录现有数据集的shape
2. 记录各个变量的type
3. 了解各个变量简单的统计信息
4. 熟悉各个变量的取值
5. 各个变量的数据质量分析（缺失值、重复值、异常值、歧义值）
6. 如果是分类问题还需要分析下正负样本比例（样本不平衡问题）

`df.info()`
`df.columns`：显示所有的变量名
`df.shape`：shape
`df.head()`：给前几个样本
`df.tail()`：给后几个样本
`df.sample(10)`：随机给几个样本
`df.describe()`：连续变量的一些描述信息，如基本统计量、分布等。
`df.describe(include=['O'])`：分类变量的一些描述信息。
`df.describe(include='all')`：全部变量的一些描述信息。
`Y_train.value_counts()`：观察取值数量

## 3.2 多变量探索
```py
# 列表汇总
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190922221630446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
```py
# 对比，直方图
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019092422214917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
```py
# 散点图
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209204304426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
```py
# 分类变量，箱图
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209204250397.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
```py
# 相关分析，热度图heatmaps1
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
# 选出和目标变量最相关的k个变量
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209205930847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

## 3.3 时间空间上的探索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200424223627828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

# 4 EDA实例
1. [跨境电商智能算法大赛-数据探索与可视化](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.7404238aQacdS0&postId=66312)
