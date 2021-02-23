'''
支持向量机
SVM.py
优点:泛化错误率低，计算开销不大，结果易解释。
缺点:对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题。


'''

from sklearn.svm import SVC

class_weights = {0:1, 1:5}#权重
kernels = ["rbf","poly","sigmoid","linear"]
c_list = [0.01, 0.1, 1, 10, 100]

model = SVC(class_weight = class_weights,kernel = kernels[0],C=C[0])
model.fit(X_train, y_train)

pred = model.predict(X_test)
