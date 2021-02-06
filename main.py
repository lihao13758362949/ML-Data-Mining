# 1 EDA 输入：文件 输出：train,test

# 2 数据预处理 输入：train,test 输出：train_pro.csv,test_pro.csv
# 3 特征工程 输入：train_pro.csv,test_pro.csv 输出：train_final.csv,test_final.csv或 train_for_tree.csv,test_for_tree.csv,train_for_lr.csv
# 4. model前准备 输入：train_final.csv,test_final.csv
feature = [x for x in train.columns if x not in ['187']]

y_train = train['187']
X_train = train[feature].values
X_test = test[feature].values
## model_selection 输入：训练集X和y 输出：splits
## metrics 输出：metrics


# 5. model #输入：splits,metrics,X_test 输出：score（cv_score或）,res,模型（feature_importance，best_iteration）
## model_params
