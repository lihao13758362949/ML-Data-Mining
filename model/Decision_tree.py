'''
Decision_tree.py
决策树
优点:计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
缺点:可能会产生过度匹配问题。

伪代码：

输入:训练集D = {(x1, y1), (x2, 2)... (xm, ym)};属性集A= {a1, a2,...,ad}.
过程:函数TreeGenerate(D, A)
1:生成结点node;
2:if D中样本全属于同一类别C then
3:    将node标记为C类叶结点; return   # 递归返回,情形(1). 当前结点包含的样本全属于同一类别，无需划分;
4:    end if
5:if A=空集 OR D中样本在A上取值相同 then
6:    将node标记为叶结点,其类别标记为D中样本数最多的类; return   # 递归返回,情形(2).当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
7:    end if
8:从A中选择最优划分属性a*; # ID3、C4.5、CART
9:for a*的每一个值av do
10:   为node生成一个分支;令Dv表示D中在a*上取值为av的样本子集;
11:   if Dv 为空 then
12:       将分支结点标记为叶结点,其类别标记为D中样本最多的类; return  # 递归返回,情形(3).当前结点包含的样本集合为空,不能划分
13:   else
14:       以TreeGenerate(Dv, A \ {a*})为分支结点
15:   end if
16: end for
输出:以node为根结点的一棵决策树







'''
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
