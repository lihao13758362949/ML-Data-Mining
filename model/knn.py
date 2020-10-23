import pandas as pd
data = pd.DataFrame({'name':['北京遇上西雅图','喜欢你','疯狂动物城','战狼2','力王','敢死队'],
                  'fight':[3,2,1,101,99,98],
                  'kiss':[104,100,81,10,5,2],
                  'type':['Romance','Romance','Romance','Action','Action','Action']})
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()  
knn.fit(data[['fight','kiss']], data['type'])
print('预测电影类型为:', knn.predict([[18, 90]]))
