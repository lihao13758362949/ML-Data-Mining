from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
