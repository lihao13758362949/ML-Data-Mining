# 6 最小角度回归（Least Angle Regression）
# 7 LARS Lasso
lasso lars是一个使用lars算法实现的lasso模型，与基于坐标下降的实现不同，它产生精确的解，它是作为其系数范数函数的分段线性。

# 8 正交匹配追踪（Orthogonal Matching Pursuit (OMP)）
# 9 贝叶斯回归（Bayesian Regression）
高斯过程
贝叶斯岭回归（Bayesian Ridge Regression）
Automatic Relevance Determination - ARD
`.BayesianRidge`

# 11 随机梯度下降（Stochastic Gradient Descent - SGD）
1. 可用于分类：
>多元分类采用OVR（One vs Rest）策略。
2. 可用于回归：
3. 时间复杂度：
>如果x是一个大小矩阵（n，p），则训练的成本为O(knp')，其中k是迭代次数（epoch），p’是每个样本的非零属性的平均数。
4. 数学原理：
>损失函数：$E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w)$
>迭代原理：$w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}+ \frac{\partial L(w^T x_i + b, y_i)}{\partial w})$
# 12 Passive Aggressive Algorithms
# 13 稳健回归（Robustness regression）
稳健回归旨在在存在损坏数据的情况下拟合回归模型：如异常值或模型中的错误。
# 14 Huber Regression
# 15 多项式回归（Polynomial regression）
用于处理非线性回归问题
类泰勒公式eshi
# 16 probit模型

# 17 sklearn.linear_model实现广义线性模型

6. `.MultiTaskLasso`
7. `.ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')`
>`l1_ratio`：即$\rho$，The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
8. `.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, positive=False, random_state=None, selection='cyclic')`
9. `.MultiTaskElasticNet`
10. `.LassoLars`
11. `.OrthogonalMatchingPursuit` 或 `.orthogonal_mp`
13. `.ARDRegression`
15. `.SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)`
>`loss`：分类损失函数： ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’；回归损失函数：‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
>`penalty`：惩罚项，默认l2
>`max_iter`：最大迭代次数。
>`verbose`：需要评价函数，若为True则表示每次迭代都输出评价指标，若为整数则每多少次迭代输出一次指标。
16. `.SGDRegressor(loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate=’invscaling’, eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)`
>`loss`：略。
17. `.Perceptron`
18. `.PassiveAggressiveClassifier`
19. `.HuberRegressor` 


5. `clf.decision_function`：决策函数 

## 17.3 可视化模型训练结果
```py
# 观察系数
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200222102303269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
```py
# 观察残差
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200222102700240.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
