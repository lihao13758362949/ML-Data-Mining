# 输入：X_train,y_train
# 输出：splits

from sklearn import model_selection

# 1.留出法（hold-out)
X_train, X_test, y_train, y_test = model_selection.train_test_split(data,target, test_size=0.4, random_state=0,stratify=None)
# test_size：测试集的比例
# n_splits：k值，进行k次的分割
# stratify：指定分层抽样变量，按该变量的类型分布分层抽样。
# .ShuffleSplit(n_splits=10, test_size=None, train_size=None, random_state=None)：打乱后分割
# .StratifiedShuffleSplit(n_splits=10, test_size=None, train_size=None, random_state=None)





# 2.自助采样法(bootstrap sampling)
#从数据集D中随机抽取一个样本，把它拷贝到训练集后放回数据集D，重复此动作m次，我们就得到了训练集$D'$，而未选中的样本就作为验证集。显然有一部分样本会出现多次，而另一部分样本不出现。
#$$\displaystyle \lim_{m\to \infty}(1-\frac{1}{m})^m= \frac{1}{e}\approx0.368$$

#即通过自助采样，D中约有36.8%的样本不会出现在$D'$中。





# 3. 交叉验证（cross-validation)
#Validation function

# n_folds = 5    k折的k,’warn’
# shuffle = False    是否打乱，默认否。
# random_state = 2021

kf = KFold(n_splits=n_folds,shuffle=shuffle,random_state=random_state)
kf = StratifiedKFold(n_splits=n_folds,shuffle=shuffle,random_state=random_state)
splits=kf.split(X_train,y_train)

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# .cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=’warn’, n_jobs=None, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’)
#cv：当cv为整数时默认使用kfold或分层折叠策略，如果估计量来自ClassifierMixin，则使用后者。另外还可以指定其它的交叉验证迭代器或者是自定义迭代器。
#scoring：指定评分方式，详见https://blog.csdn.net/weixin_42297855/article/details/99212685
def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "mean_squared_error", cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = "mean_squared_error", cv = 10))
    return(rmse)



# 4 <学习曲线>
from sklearn.model_selection import learning_curve, validation_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt  
plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)  

# 5 <调参>
## 5.1 <贪心调参>
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score
sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])

## 5.2 <Grid Search 调参>网格搜索
from sklearn.model_selection import GridSearchCV

parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
clf.best_params_
# {'max_depth': 15, 'num_leaves': 55, 'objective': 'regression'}
model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)
np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
## 5.3 <随机搜索>

## 5.4 <贝叶斯优化算法>
'''
贝叶斯优化算法在寻找最优最值参数时，采用了与网格搜索、随机搜
索完全不同的方法。网格搜索和随机搜索在测试一个新点时 ，会忽略前一
个点的信息;而贝叶斯优化算法则充分利用了之前的信息。贝叶斯优化算
法通过对目标函数形状进行学习，找到使目标函数向全局最优值提升的参
数。具体来说，它学习目标函数形状的方法是，首先根据先验分布，假设
-个搜集函数;然后，每一次使用新的采样点来测试目标函数时,利用这
个信息来更新目标函数的先验分布;最后，算法测试由后验分布给出的全
局最值最可能出现的位置的点。对于贝叶斯优化算法，有一个需要注意的
地方，一但找到了一-个局部最优值，它会在该区域不断采样，所以很容易
陷入局部最优值。为了弥补这个缺陷，贝叶斯优化算法会在探索和利用之
间找到-一个平衡点，“ 探索”就是在还未取样的区域获取采样点;而“利
用”则是根据后验分布在最可能出现全局最值的区域进行采样。

'''
from bayes_opt import BayesianOptimization
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val
rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
rf_bo.maximize()





