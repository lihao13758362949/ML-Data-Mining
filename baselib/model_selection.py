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
