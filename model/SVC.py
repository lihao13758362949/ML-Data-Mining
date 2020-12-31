from sklearn.svm import SVC

class_weights = {0:1, 1:5}#权重
kernels = ["rbf","poly","sigmoid","linear"]
c_list = [0.01, 0.1, 1, 10, 100]

model = SVC(class_weight = class_weights,kernel = kernels[0],C=C[0])
model.fit(X_train, y_train)

pred = model.predict(X_test)
