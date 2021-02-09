

>**集成学习的两大问题**：每一轮如何改变训练数据的权值？如何将弱分类器组合成一个强分类器？


# 2 Bagging
独立的训练一些基学习器(一般倾向于强大而复杂的模型比如完全生长的决策树)，然后综合他们的预测结果。
## 2.1 Bagging
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

## 2.2 随机森林（Random Forest）
>随机森林在基学习器较少的时候表现不太好，但随着基学习器数目的增加，随机森林通常会收敛到更低的方差。

和决策树算法类似，先从候选划分属性中随机选取$k=log_2d$（推荐）个属性，接着用划分算法选择最优的属性，构建基决策树们。然后做法和bagging相同，用简单平均（连续）或投票法（离散）得到最终学习器。
>极端随机森林即k=1

# 3 学习法（Stacking）
训练几个初级学习器，然后用他们的预测结果来训练次级（元）学习器。

# 4 Boosting
## 4.1 AdaBoost
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114638648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114652429.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)


## 4.2 boosting tree（提升树）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114610158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)

## 4.3 Gradient Tree Boosting （GB\(R\)T，梯度提升树）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114555628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)


## 4.4 XGBoost
## 4.5 LightGBM


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200425092947847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

