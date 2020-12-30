***
# 0 写在前面
参考资料：
1. [pandas分批读取大数据集](https://blog.csdn.net/htbeker/article/details/86542412)
2. [智慧海洋建设baseline——wbbhcb](https://github.com/wbbhcb/zhhyjs_baseline)
3. [（回归问题）Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
4. [（分类问题）Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)





在分析数据的过程中，还必须要弄清楚的以下数据相关的问题：

1. 数据量是否充分，是否有外部数据可以进行补充；
5. 训练集和测试集的数据分布是否有差异;
# 2 读取数据


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

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209205930847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

## 3.3 时间空间上的探索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200424223627828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

# 4 EDA实例
1. [跨境电商智能算法大赛-数据探索与可视化](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.7404238aQacdS0&postId=66312)
