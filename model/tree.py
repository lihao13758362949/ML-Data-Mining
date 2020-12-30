from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 参数
class_weights = {1:1, 2:5}

clf = DecisionTreeClassifier(min_samples_leaf = 6 ,class_weight=class_weights)
clf.fit(X_train, y_train)
#DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
#DecisionTreeRegressor(criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)


#clf.predict_proba([[2., 2.]])#预测每个类的概率，即叶中相同类的训练样本的分数
pred = clf.predict(X_test)


def get_leaf(train_x, train_y, val_x):
    from sklearn.tree import DecisionTreeClassifier
    train_x, train_y, val_x = map(np.array, [train_x, train_y, val_x])
    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)
    val_x = val_x.reshape(-1, 1)
    m = DecisionTreeClassifier(min_samples_leaf=0.001, max_leaf_nodes=25)
    m.fit(train_x, train_y)
    return m.apply(val_x)
