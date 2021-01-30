# 1 EDA 输出train,test

# 2 数据预处理 输入：train,test 输出：train,test

# 3. model前准备 输入：train,test
feature = [x for x in train.columns if x not in ['187']]

y_train = train['187']
X_train = train[feature].values
X_test = test[feature].values
## model_selection 输入：训练集X和y 输出：splits
## metrics 输出：metrics


# 4. model #输入：splits,metrics,X_test
## model_params
