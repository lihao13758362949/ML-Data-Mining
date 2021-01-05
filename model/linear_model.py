import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# 数据导入（这里以波士顿房价为例）
loaded_data = datasets.load_boston() 
X_data = loaded_data.data 
y_data = loaded_data.target 
print(X_data.shape)  # (506, 13)
print(y_data.shape)  # (506,)
X=pd.DataFrame(X_data)
y=pd.DataFrame(y_data)

# 模型训练、结果分析
def LR(X,y,type=1,test_size=0.3,cv=10,random_state=2021,plot=False):
      # 线性回归
      # type:1表示留出法hold-out,2表示交叉验证法,test_size和cv是他们的参数
      
      lr = linear_model.LinearRegression()
        
      if type==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            # 输出指标
            print('Mean squared error: %.2f'
                % mean_squared_error(y_test, y_pred))
            print('Coefficient of determination: %.2f'
                % r2_score(y_test, y_pred))# 此句等价于lr.score(X_test, y_test)
        
      if type==2:
            y_pred = cross_val_predict(lr, X, y, cv=10)
            
      print("Intercept value",lr.intercept_) #截距
      print(pd.DataFrame(list(zip(list(X.columns),lr.coef_.flatten().tolist())),columns=["特征","系数"]))


      # Look at predictions on training and validation set
      print("RMSE on Training set :", rmse_cv_train(lr).mean())
      print("RMSE on Test set :", rmse_cv_test(lr).mean())
      y_train_pred = lr.predict(X_train)
      y_test_pred = lr.predict(X_test)

      # Plot residuals
      plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
      plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
      plt.title("Linear regression")
      plt.xlabel("Predicted values")
      plt.ylabel("Residuals")
      plt.legend(loc = "upper left")
      plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
      plt.show()

      # Plot predictions
      if plot==True:
            plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
            plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
            plt.title("Linear regression")
            plt.xlabel("Predicted values")
            plt.ylabel("Real values")
            plt.legend(loc = "upper left")
            plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
            plt.show()
            
            # 画拟合线
            plt.scatter(X,y,color='blue')
            plt.plot(X,lr.predict(X),color='red',linewidth=4)
            plt.show()
def Ridge_model_train(X,y,alpha=0.05,type=1,test_size=0.3,cv=10):
      # 岭回归 L2-penalty
      # alpha：超参数，惩罚项系数
      
      # 常规用法
      reg = linear_model.Ridge(alpha=.5)
      reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

      reg.coef_

      reg.intercept_
      reg.predict([2,2])
      ## 搜索超参数1
      alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
      cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
      cv_ridge = pd.Series(cv_ridge, index = alphas)
      cv_ridge.plot(title = "Validation - Just Do It")
      plt.xlabel("alpha")
      plt.ylabel("rmse")
      
      ## 搜索超参数2
      ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
      ridge.fit(X_train, y_train)
      alpha = ridge.alpha_
      print("Best alpha :", alpha)

      print("Try again for more precision with alphas centered around " + str(alpha))
      ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                                alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                                alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                      cv = 10)
      ridge.fit(X_train, y_train)
      alpha = ridge.alpha_
      print("Best alpha :", alpha)
      
      ridge= linear_model.Ridge(alpha=alpha)  # 设置lambda值
      ridge.fit(X,y)  #使用训练数据进行参数求解
      Y_hat1 = ridge.predict(X_test)  #对测试集的预测
      
scorer = make_scorer(mean_squared_error, greater_is_better = False)
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring=scorer, cv = 5))
    return(rmse)

def Lasso_model_train(X,y,alpha=0.05,type=1,test_size=0.3,cv=10):
      # 岭回归 L1-penalty
      # alpha：超参数，惩罚项系数
      lasso= linear_model.Lasso(alpha=alpha)  # 设置lambda值
      lasso.fit(X,y)  #使用训练数据进行参数求解
      Y_hat2=lasso.predict(X_test)#对测试集预测
      
      model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
      #rmse_cv(model_lasso).mean()
      coef = pd.Series(model_lasso.coef_, index = X_train.columns)
      print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
      imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
      #matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
      #imp_coef.plot(kind = "barh")
      #plt.title("Coefficients in the Lasso Model")

def ElasticNet_model_train(X,y,alpha=0.05,l1_ratio=0.4,type=1,test_size=0.3,cv=10):
      # 岭回归 L1+L2-penalty
      # alpha：超参数，惩罚项系数
      elastic= linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)  # 设置lambda值,l1_ratio值
      elastic.fit(X,y)  #使用训练数据进行参数求解
      y_hat3 = elastic.predict(X_test)  #对测试集的预测

#def linear_model_outputs:
      
# .LogisticRegression

# 可视化
def linear_model_plot(X,y,type=1)
      # type:1为散点图+回归直线，2为预测偏差图
      if type==1:
            plt.scatter(X, y_test,  color='blue')
            plt.plot(X_test, diabetes_y_pred, color='red', linewidth=3)
            plt.show()
      if type==2:
            plt.scatter(y_data, predicted, color='y', marker='o')
            plt.scatter(y_data, y_data,color='g', marker='+')
            plt.show()
