'''
lgb.py
LightGBM
'''

# 1 <载入数据>
# >LightGBM将数据存储在Dataset对象里
# >支持的数据类型：
# >libsvm/tsv/csv/txt format file
# >NumPy 2D array(s), pandas DataFrame, H2O DataTable’s Frame, SciPy sparse matrix
# >LightGBM binary file

lgb.Dataset(data, label=None, reference=None, weight=None, group=None, init_score=None, silent=False, feature_name='auto', categorical_feature='auto', params=None, free_raw_data=True)
''' 
reference：如果这是用于验证的数据集，则应作为reference
'''

# 2 <设置参数>
param = {'num_leaves': 31, 'objective': 'binary',...}
'''
参数介绍：
'num_leaves': 31
'objective': 'binary'
'metric':'auc'
'max_depth'
‘task’: ‘train’,
‘boosting_type’: ‘gbdt’, # 设置提升类型
‘objective’: ‘regression’, # 目标函数
‘metric’: {‘l2’, ‘auc’}, # 评估函数
‘num_leaves’: 31, # 叶子节点数
‘learning_rate’: 0.05, # 学习速率
‘feature_fraction’: 0.9, # 建树的特征选择比例
‘bagging_fraction’: 0.8, # 建树的样本采样比例
‘bagging_freq’: 5, # k 意味着每 k 次迭代执行bagging
‘verbose’: 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
'''

# 3 <训练和预测>
lgb.train(params, train_set, num_boost_round=100, valid_sets=None, valid_names=None, fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, evals_result=None, verbose_eval=True, learning_rates=None, keep_training_booster=False, callbacks=None)
'''
num_boost_round：boost迭代次数
evals：一对对 (DMatrix, string)组成的列表，培训期间将评估哪些指标的验证集列表。验证指标将帮助我们跟踪模型的性能。用evallist = [(dtest, 'eval'), (dtrain, 'train')]指定。
valid_sets：设置验证指标数据集，如valid_sets = [trn_data, val_data]。
valid_names：valid_sets的名字。
early_stopping_rounds：验证指标需要至少在每轮early_stopping_rounds中改进一次才能继续训练，例如early_stopping_rounds=200表示每200次迭代将会检查验证指标是否有改进，如果没有就会停止训练，如果有多个指标，则只判断最后一个指标
verbose_eval：取值可以是bool型也可以是整数，当取值为True时，表示每次迭代都显示评价指标，当取值为整数时，表示每该取值次数轮迭代后显示评价指标
'''
lgb.cv(params, train_set, num_boost_round=100, folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, fpreproc=None, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, eval_train_metric=False)

clf.predict(X, raw_score=False, num_iteration=None, pred_leaf=False, pred_contrib=False, **kwargs)
'''
num_iteration：int或None，限制预测中的迭代次数。如果None且存在最佳迭代，则使用它；否则，使用所有树。如果<=0，则使用所有树（无限制）。常用num_iteration=clf.best_iteration。
'''
# 6 <实例>
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#数据略

param = {'boosting_type': 'gbdt',
         'num_leaves': 20,
         'min_data_in_leaf': 20, 
         'objective':'regression',
         'max_depth':6,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2020)
oof_lgb = np.zeros(len(X_train_))
predictions_lgb = np.zeros(len(X_test_))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
   # print(trn_idx)
   # print(".............x_train.........")
   # print(X_train[trn_idx])
  #  print(".............y_train.........")
  #  print(y_train[trn_idx])
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
# oof_lgb_final = np.argmax(oof_lgb, axis=1)  
print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train_)))
# pred_label = np.argmax(prediction, axis=1)


# 7 <一体化函数>
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(x_train, y_train)

lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split', **kwargs)
'''
boosting_type：‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
num_leaves：最大的叶子数，即基学习器数量
max_depth：基学习器的最大树深度，<=0表示没有限制
learning_rate
n_estimators
objective：‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.
'''
