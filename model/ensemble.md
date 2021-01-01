

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
能够降低模型的bias，迭代地训练 Base Model，每次根据上一个迭代中预测错误的情况修改训练样本的权重。也即 Gradient Boosting 的原理。比 Bagging 效果好，但更容易 Overfit。
## 4.1 AdaBoost[Freund and Schapire,1997]
>提高前一轮弱分类器错误分类样本的权值，降低被正确分类样本的权值。
>**加权投票法**组合分类器

**AdaBoost算法**：
输入：训练数据集$T={(x_1,y_1),...,(x_m,y_m)}$，其中$x_i\in \mathbb{R}^n,y_i\in \{-1,+1\}$；弱学习器算法。
输出：最终分类器$G(x)$。
（1）初始化训练数据的权值
$$D_1=(w_{11},...,w_{1m}),w_{1i}=\frac{1}{m},i=1,2,...,m$$
（2）依次对K个弱学习器进行学习，$k=1,2,...,K$
$\quad$（a）使用相同权值分布$D_k$的训练数据集学习，得到基本分类器
$$G_k(x)\to\{-1,+1\}$$
$\quad$（b）计算$G_k(x)$在训练数据集上的分类误差率，删除$e_k\ge \frac{1}{2}$的基学习器，说明基学习器比随机猜测还差。
$$\displaystyle e_k=P(G_k(x_i)=\not y_i)=\sum_{i=1}^mw_{ki}I(G_k(x_i)=\not y_i)$$
$\quad$（c）计算$G_k(x)$的系数
$$\alpha_k=\frac{1}{2}ln\frac{1-e_k}{e_k}$$
$\quad$（d）更新训练数据的权值
$$D_{k+1}=(w_{k+1,1},...,w_{k+1,m}),w_{k+1,i}=\frac{w_{ki}}{Z_k}e^{-\alpha_ky_iG_k(x_i)},i=1,2,...,m$$
这里，$Z_k$是规范化因子$\displaystyle Z_k=\sum^m_{i=1}w_{ki}e^{-\alpha_ky_iG_k(x_i)}$，它使$D_{k+1}$成为一个概率分布。
（3）构建基本分类器的线性组合
$$\displaystyle f(x)=\sum_{k=1}^K\alpha_kG_k(x)$$
得到最终分类器：
$$G(x)=sign(f(x))$$
参考：《统计学习方法》李航
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114638648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114652429.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)

## 4.2 boosting tree（提升树）
**提升树算法**（向前分布算法，逐渐减少残差）
**注：提升树算法仅在损失函数为平方误差损失函数时适用。**
输入：训练数据集$T={(x_1,y_1),...,(x_m,y_m)}$，其中$x_i\in \mathbb{R}^n,y_i\in \{-1,+1\}$。
输出：提升树$f_K(x)$。
（1）初始化$f_0(x)=0$
（2）对K棵决策树，$k=1,2,...,K$
$\quad$（a）计算残差：$r_{ki}=y_i-f_{k-1}(x_i),i=1,2,...,m$
$\quad$（b）拟合残差学习一个回归树：$T(x;\Theta_k)$
$\quad$（c）更新：$f_k(x)=f_{k-1}(x)+T(x;\Theta_k)$
（3）得到回归问题的提升树（分类问题即对回归问题的提升树进行**符号函数**变换）：
$$f_K(x)=\sum_{k=1}^KT(x;\Theta_k)$$
参考：《统计学习方法》李航
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114610158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)

## 4.3 Gradient Tree Boosting （GB\(R\)T，梯度提升树）
**梯度提升树算法**（一般化的提升树算法）
输入：训练数据集$T={(x_1,y_1),...,(x_m,y_m)}$，其中$x_i\in \mathbb{R}^n,y_i\in \{-1,+1\}$；损失函数$L(y,f(x))$。
输出：提升树$\hat f(x)$。
（1）初始化$f_0(x)=arg min_c\sum^m_{i=1}L(y_i,c)$
（2）对K棵决策树，$k=1,2,...,K$
$\quad$（a）计算损失函数的负梯度：$r_{ki}=-\frac{\partial L(y_i,f_{k-1}(x_i))}{\partial f_{k-1}(x_i)},i=1,2,...,m$
$\quad$（b）对$r_{ki}$拟合一个回归树，得到第K棵树的叶结点区域：$R_{kj},j=1,2,...J$
$\quad$（c）对$j=1,2,...,J$，计算$\displaystyle c_{kj}=argmin_c\sum_{x_i\in R_{kj}}L(y_i,f_{k-1}(x_i)+c)$
$\quad$（d）更新：$f_k(x)=f_{k-1}(x)+\sum_{j=1}^Jc_{kj}I(x\in R_{kj})$
（3）得到回归问题的提升树（分类问题即对回归问题的提升树进行**符号函数**变换）：
$$\displaystyle \hat f(x)=f_K(x)=\sum_{k=1}^K\sum_{j=1}^Jc_{kj}I(x\in R_{kj})$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201023114555628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70#pic_center)


## 4.4 XGBoost
## 4.5 LightGBM


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200425092947847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

