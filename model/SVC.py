from sklearn import metrics
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print (metrics.classification_report(y_test, pred))
print (metrics.confusion_matrix(y_test, pred))
print (metrics.accuracy_score(y_test, pred))
