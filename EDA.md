***
# 0 写在前面
参考资料：
1. [pandas分批读取大数据集](https://blog.csdn.net/htbeker/article/details/86542412)


# 2 读取数据
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

#  3 数据探索

## 3.3 时间空间上的探索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200424223627828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

