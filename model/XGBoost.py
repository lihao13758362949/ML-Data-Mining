'''
XGBoost.py
'''

import xgboost as xgb

# 1 <载入数据>
# >xgboost将数据存储在DMatrix对象里
# >支持的数据类型：
# >LibSVM text format file
# >Comma-separated values (CSV) file
# >NumPy 2D array
# >SciPy 2D sparse array
# >Pandas data frame
# >XGBoost binary buffer file.
# >注：xgb载入分类变量前要先one_hot encoding

xgb.DMatrix(data, label=None, missing=None, weight=None, silent=False, feature_names=None, feature_types=None, nthread=None)
'''
载入数据到DMatrix对象。

label：指定标签值向量/矩阵。
missing：指定缺失值在矩阵中的值。
weight：指定权重变量。
'''

DMatrix.save_binary('train.buffer') # 存储DMatrix对象，下次使用时能加快加载速度。

# 2 <模型参数>
param = {'max_depth': 2,'eta': 1, 'objective': 'binary:logistic',...}
'''
参数介绍：

'nthread':4
'eval_metric'：['auc','ams@0','rmse']
'max_depth': 2
'eta': 1
'objective':：'reg:linear'，'binary:logistic'
"booster":'gbtree'
'subsample': 0.7
'colsample_bytree': 0.8
'silent': True
'''
# 3 <训练和预测>
clf=xgb.train(params, dtrain, num_boost_round=10, evals=[], obj=None, feval=None, maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None)
'''
num_boost_round：boost迭代次数
evals：一对对 (DMatrix, string)组成的列表，培训期间将评估哪些指标的验证集列表。验证指标将帮助我们跟踪模型的性能。用evallist = [(dtest, 'eval'), (dtrain, 'train')]指定。
obj
feval：自定义评价函数
maximize
early_stopping_rounds：验证指标需要至少在每轮early_stopping_rounds中改进一次才能继续训练，例如early_stopping_rounds=200表示每200次迭代将会检查验证指标是否有改进，如果没有就会停止训练，如果有多个指标，则只判断最后一个指标
evals_result
verbose_eval：取值可以是bool型也可以是整数，当取值为True时，表示每次迭代都显示评价指标，当取值为整数时，表示每该取值次数轮迭代后显示评价指标
xgb_model
callbacks
learning_rates
'''

xgb.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None, metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True)
'''
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
'''

bst.save_model('0001.model')

ypred = clf.predict(data, output_margin=False, ntree_limit=None, validate_features=True)
'''
ntree_limit：限制预测中的树数；如果定义了最佳树数限制，则默认为最佳树数限制，否则为0（使用所有树）
'''

xgb.plot_tree(bst, num_trees=2)
xgb.to_graphviz(bst, num_trees=2)


# 6 <实例>
import xgboost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#X_train,y_train略

#自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score
    
xgb_params = {"booster":'gbtree','eta': 0.005, 'max_depth': 5, 'subsample': 0.7, 
              'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
    
    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params,feval = myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train_)))

# 7 <一体化函数>
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7)

scores_train = []
scores = []

## 5折交叉验证方式
sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
for train_ind,val_ind in sk.split(X_data,Y_data):
    
    train_x=X_data.iloc[train_ind].values
    train_y=Y_data.iloc[train_ind]
    val_x=X_data.iloc[val_ind].values
    val_y=Y_data.iloc[val_ind]
    
    xgr.fit(train_x,train_y)
    pred_train_xgb=xgr.predict(train_x)
    pred_xgb=xgr.predict(val_x)
    
    score_train = mean_absolute_error(train_y,pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y,pred_xgb)
    scores.append(score)

print('Train mae:',np.mean(score_train))
print('Val mae',np.mean(scores))
