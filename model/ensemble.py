'''
集成学习 ensemble.py
输入：splits,metrics,X_test
输出：score（cv_score或）,res,模型（feature_importance，best_iteration）

'''
from sklearn import datasets  
from sklearn import model_selection  
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np


# 1 <简单加权融合>
# 1.1 <Voting> 简单投票法
# 载入数据集
iris = datasets.load_iris()  
# 只要第1,2列的特征
x_data, y_data = iris.data[:, 1:3], iris.target  

# 定义三个不同的分类器
clf1 = KNeighborsClassifier(n_neighbors=1)  
clf2 = DecisionTreeClassifier() 
clf3 = LogisticRegression()  

sclf = VotingClassifier([('knn',clf1),('dtree',clf2), ('lr',clf3)])   
# VotingClassifier(estimators, voting=’hard’, weights=None, n_jobs=None, flatten_transform=True)
# voting：默认hard表示绝对多数投票法，即选择超过半数的票数；若为soft表示相对多数投票法，即选择最多票数。
#VotingRegressor(estimators, weights=None, n_jobs=None) 
for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN','Decision Tree','LogisticRegression','VotingClassifier']):  
  
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')  
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label)) 
   
  
  
  
  
# 1.2 <简单的平均>
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
      
def Weighted_method(test_pre1,test_pre2,test_pre3,w=[1/3,1/3,1/3]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)
    return Weighted_result        
        
# 2 <排序融合(Rank averaging)，log融合> 
        
        

      
      
# 3 <Boosting>
# >能够降低模型的bias，迭代地训练 Base Model，每次根据上一个迭代中预测错误的情况修改训练样本的权重。也即 Gradient Boosting 的原理。比 Bagging 效果好，但更容易 Overfit。
## 3.1 <AdaBoost>
# >提高前一轮弱分类器错误分类样本的权值，降低被正确分类样本的权值。
# >加权投票法组合分类器
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss=’linear’, random_state=None)
AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
'''
n_estimators：基学习器数量
base_estimator：基学习器类型，默认`.tree.DecisionTreeRegressor(max_depth=3)`
'''

## 3.2 <boosting tree>（提升树）
# >提升树算法（向前分布算法，逐渐减少残差） 
# >注：提升树算法仅在损失函数为平方误差损失函数时适用



## 3.3 <Gradient Boosting Decision Tree> （GBDT，梯度提升树）
# >一般化的提升树算法
GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
GradientBoostingRegressor(loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

## 3.4 <XGBoost>
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model
## 3.5 <LightGBM>
import numpy as np
import pandas as pd
import lightgbm as lgb

lgb_param = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'metric' : {'binary_error'},
    'num_leaves' : 120,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.7,
    'bagging_freq' : 5,
    # 'min_data_in_leaf' : 10,
    'verbose' : 0
}

cv_score = []
res = np.zeros(X_test.shape[0])

for idx,(train_idx,val_idx) in enumerate(splits):
    print('-'*50)
    print('iter{}'.format(idx+1))
    X_trn,y_trn = X_train[train_idx],y_train[train_idx]
    X_val,y_val = X_train[val_idx],y_train[val_idx]

    dtrn = lgb.Dataset(X_trn,label=y_trn)
    dval = lgb.Dataset(X_val, label=y_val)

    bst = lgb.train(lgb_param,dtrn,500000,valid_sets=dval,
                    early_stopping_rounds=150,verbose_eval=50)
    preds = bst.predict(X_val,num_iteration=bst.best_iteration)
    lgb_pred[val_idx] = preds

    preds = transform(preds)

    score = accuracy_score(y_val,preds)
    print(score)
    cv_score.append(score)
    res += transform(bst.predict(X_test,num_iteration=bst.best_iteration)) # 输出时要改为0或1

    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = train[feature].columns
    dfFeature['score'] = bst.feature_importance()
    dfFeature = dfFeature.sort_values(by='score',ascending=False)
    dfFeature.to_csv('featureImportance.csv',index=False,sep='\t')
    print(dfFeature)
    
print('offline mean ACC score:',np.mean(cv_score))
    
def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm
    
    

3. `.BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)`
4. `.BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)`

7. `.ExtraTreesClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)`
8. `.ExtraTreesRegressor(n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)`

13. `.HistGradientBoostingClassifier(loss=’auto’, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=256, scoring=None, validation_fraction=0.1, n_iter_no_change=None, tol=1e-07, verbose=0, random_state=None)`：数据量较大时效果比`GradientBoostingClassifier`好得多。
14. `.HistGradientBoostingRegressor(loss=’least_squares’, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=256, scoring=None, validation_fraction=0.1, n_iter_no_change=None, tol=1e-07, verbose=0, random_state=None)`




# 4 <Bagging>
# >独立的训练一些基学习器(一般倾向于强大而复杂的模型比如完全生长的决策树)，然后综合他们的预测结果。
"""
通常为了获得差异性较大的基学习器，我们对不同的基学习器给不同的训练数据集。根据**采样方式**有以下变体：
Pasting:直接从样本集里随机抽取的到训练样本子集
Bagging:自助采样(有放回的抽样)得到训练子集
Random Subspaces:列采样,按照特征进行样本子集的切分
Random Patches:同时进行行采样、列采样得到样本子集

当训练了许多基学习器后，将他们加权平均（连续）或投票法（离散）得到最终学习器。

这里给出投票法的几种类型：
绝对多数投票法：如果标记投票超过半数则预测标记，否则拒绝预测。
相对多数投票法：预测为得票最多的标记，若有多个得票相同，则随机选取一个。
加权投票法：以学习器的准确率为权重加权投票，并选择最多的票数标记。
"""
## 4.1 <Random Forest 随机森林>
# >随机森林的优点：防过拟合、抗噪声、无需规范化、速度快、
# >随机森林的缺点：某些噪声过大的问题上会过拟合、对取值多的变量有偏好，因此属性权值不可信
# >随机森林在基学习器较少的时候表现不太好，但随着基学习器数目的增加，随机森林通常会收敛到更低的方差。
# >和决策树算法类似，先从候选划分属性中随机选取$k=log_2d$（推荐）个属性，接着用划分算法选择最优的属性，构建基决策树们。然后做法和bagging相同，用简单平均（连续）或投票法（离散）得到最终学习器。
# >极端随机森林即k=1


# `.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)`
# `.RandomForestRegressor(n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)`
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train) 
# 随机森林的可解释性
importances = clf.feature_importances_
#计算随机森林中所有的树的每个变量的重要性的标准差 
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
#按照变量的重要性排序后的索引 
indices = np.argsort(importances)[::-1]

### 绘图过程
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="c", yerr=std[indices], align="center")
plt.xticks(fontsize=14)
plt.xticks(range(X.shape[1]), df.columns.values[:-1][indices],rotation=40)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
     
      

    
  
  
  
  
# 5 <Stacking>
# >先训练初级学习器，然后用预测值来训练次级学习器
# >训练几个初级学习器，然后用他们的预测结果来训练次级（元）学习器。

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)
      
for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):
 
    scores = model_selection.cross_val_score(clf, X, y,cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))

    
from sklearn import linear_model
# 将lgb和xgb和ctb的结果进行stacking
train_stack = np.vstack([oof_lgb,oof_xgb,oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb,predictions_cb]).transpose()


folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]
    
    clf_3 = linear_model.BayesianRidge()
    #clf_3 =linear_model.Ridge()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
    
print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train_)))    
    
# <用基学习器的预测结果作为输入训练二级学习器>
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    """ author：Cookly """

    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
        """
        Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_fold stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
          early_stopping_rounds : early_stopping_rounds.
        """
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        """ fit model. """
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.K = KFold(n_splits=self.stacking_num, shuffle=True, random_state=927)
            for k, (train_index, test_index) in enumerate(self.K.split(X)):
                print(f'******************* 第{k + 1}次stacking开始喽(●• ̀ω•́ )✧ *******************')
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_test = X.iloc[test_index]
                y_test = y.iloc[test_index]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(self.params,
                                lgb_train,
                                num_boost_round=self.num_boost_round,
                                valid_sets=lgb_eval,
                                categorical_feature=cat_cols,
                                feval=rmsle,
                                early_stopping_rounds=self.early_stopping_rounds,
                                verbose_eval=0)

                self.stacking_model.append(gbm)

                pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                layer_train[test_index, 1] = pred_y
            stack_pred = pd.DataFrame({'stack_pred': layer_train[:, 1]}, index=X.index)
            X = pd.concat([X, stack_pred], axis=1)
            # X = np.hstack((X, layer_train[:,1].reshape((-1,1))))
            print('stacking finish')
        else:
            pass
        for bn in range(self.bagging_num):
            print(f'******************* 第{bn + 1}次bagging开始喽(●• ̀ω•́ )✧ *******************')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)
            y_train = pd.Series(y_train, name=y.name)
            y_test = pd.Series(y_test, name=y.name)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=self.num_boost_round,
                            valid_sets=lgb_eval,
                            categorical_feature=cat_cols,
                            feval=rmsle,
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval=0)

            self.bagging_model.append(gbm)
            print('bagging finish')

    def predict(self, X_pred):
        """ predict test data. """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                print(f'******************* 第{sn + 1}次stacking predict(●• ̀ω•́ )✧ *******************')
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                test_pred[:, sn] = pred
            # X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1,1))))
            stack_pred = pd.DataFrame({'stack_pred': test_pred.mean(axis=1)}, index=X_pred.index)
            X_pred = pd.concat([X_pred, stack_pred], axis=1)
        else:
            pass
        for bn, gbm in enumerate(self.bagging_model):
            print(f'******************* 第{bn + 1}次bagging predict(●• ̀ω•́ )✧ *******************')
            pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred
        print('predict finish')
        return pred_out / self.bagging_num

###########################################################################

# sbbtree.fit(X, y)
# pred = sbbtree.predict(X_test)