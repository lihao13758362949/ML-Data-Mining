```
kNN.py
k近邻算法
有监督分类算法
优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高

伪代码：
对未知类别属性的数据集中的每个点依次执行以下操作:
(1)计算已知类别数据集中的点与当前点之间的距离;
(2)按照距离递增次序排序;
(3)选取与当前点距离最小的k个点;
(4)确定前k个点所在类别的出现频率;
(5)返回前k个点出现频率最高的类别作为当前点的预测分类。

```










import pandas as pd
data = pd.DataFrame({'name':['北京遇上西雅图','喜欢你','疯狂动物城','战狼2','力王','敢死队'],
                  'fight':[3,2,1,101,99,98],
                  'kiss':[104,100,81,10,5,2],
                  'type':['Romance','Romance','Romance','Action','Action','Action']})
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()  
knn.fit(data[['fight','kiss']], data['type'])
print('预测电影类型为:', knn.predict([[18, 90]]))





# <以下是手推代码>
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
    
group,labels = createDataSet()

def classify0(inX,dataSet,labels,k):
    '''
    inX为要分类的输入数据
    dataSet为训练样本集
    labels为标签向量
    k为邻居数目
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    
    return sortedClassCount[0][0]
