import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split，cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# 数据导入（这里以波士顿房价为例）
loaded_data = datasets.load_boston() 
x_data = loaded_data.data 
y_data = loaded_data.target 
print(shape(x_data))  # (506, 13)
print(shape(y_data))  # (506,)
print(x_data[:2, :])
print(y_data[:2]) 


# 模型训练、结果分析
def LR_model_train(X,y,type=1,test_size=0.3,cv=10):
      # 线性回归
      # type:1表示留出法hold-out,2表示交叉验证法,test_size和cv是他们的参数
      
      model = linear_model.LinearRegression()
      if type==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=2020)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
      if type==2:
            y_pred = cross_val_predict(model, X, y, cv=10)
      print("Intercept value",model.intercept_) #截距
      print(pd.DataFrame(list(zip(list(X_data.columns),model.coef_.flatten().tolist())),columns=["特征","系数"]))
      
      # 输出指标
      print('Mean squared error: %.2f'
            % mean_squared_error(y_test, y_pred))
      print('Coefficient of determination: %.2f'
            % r2_score(y_test, y_pred))# 此句等价于regr.score(X_test, y_test)
      return predictions

def Ridge_model_train(X,y,alpha=0.05,type=1,test_size=0.3,cv=10):
      # 岭回归 L2-penalty
      # alpha：超参数，惩罚项系数
      
      
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

def Lasso_model_train(X,y,alpha=0.05,type=1,test_size=0.3,cv=10):
      # 岭回归 L1-penalty
      # alpha：超参数，惩罚项系数
      lasso= linear_model.Lasso(alpha=alpha)  # 设置lambda值
      lasso.fit(X,y)  #使用训练数据进行参数求解
      Y_hat2=lasso.predict(X_test)#对测试集预测

def ElasticNet_model_train(X,y,alpha=0.05,l1_ratio=0.4,type=1,test_size=0.3,cv=10):
      # 岭回归 L1+L2-penalty
      # alpha：超参数，惩罚项系数
      elastic= linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)  # 设置lambda值,l1_ratio值
      elastic.fit(X,y)  #使用训练数据进行参数求解
      y_hat3 = elastic.predict(X_test)  #对测试集的预测

#def linear_model_outputs:
      


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
