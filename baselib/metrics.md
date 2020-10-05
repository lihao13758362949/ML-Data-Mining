# 分类器评价指标
7. $F_\beta$：$F_\beta=\displaystyle\frac{(1+\beta^2)\times precision \times recall} {\beta^2\times precision + recall}$，加权调和平均，考虑了偏好，$\beta>1$时查全率更偏好。
8. Micro-F1：多类别问题中的F1指标，计算出所有类别总的Precision和Recall，然后计算F1。
9. Macro-F1：多类别问题中的F1指标，计算出每一个类的Precison和Recall后计算F1，最后将F1平均。



# 聚类器评价指标
## 2.1 外部指标
$\begin{array}{l}{\left.a=|S S|, \quad S S=\left\{\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\}} \\ {\left.b=|S D|, \quad S D=\left\{\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right)\right\}} \\ {\left.c=|D S|, \quad D S=\left\{\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\}} \\ {\left.d=|D D|, \quad D D=\left\{\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right)\right\}}\end{array}$
其中$(x_i,x_j)$表示一对样本，将他们放到聚类模型$\lambda$和参考模型$\lambda^\star$中有四种情况。
由于每个样本对$(x_i,x_j)$ $(i < j)$仅能出现在一个集合中，因此有a+b+c+d= m(m- 1)/2成立。
### 2.1.1 Jaccard系数(Jaccard Coefficient, 简称JC)
$\mathrm{JC}=\frac{a}{a+b+c}$
[0,1]，越大越好
### 2.1.2 FM指数(Fowlkes and Mallows Index,简称FMI)
$\mathrm{FMI}=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$
[0,1]，越大越好
### 2.1.3 Rand指数(Rand Index,简称RI)
用于评价相似度。
$\mathrm{RI}=\frac{2(a+d)}{m(m-1)}$
[0,1]，越大越好
### 2.1.4 调整Rand指数（ARI）
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
[-1,1]
### 2.1.5 互信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191016130839829.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191016130851328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
### 2.1.6 同质性、完备性与v-测度
[0,1]，越大越好
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019101613133873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
### 2.1.7 Fowlkes-Mallows scores
$\text{FMI} = \frac{\text{TP}}{\sqrt{(\text{TP} + \text{FP}) (\text{TP} + \text{FN})}}$
### 2.1.8 轮廓系数
a表示聚类正确，b表示聚类错误。则轮廓系数$s = \frac{b - a}{max(a, b)}$
### 2.1.9 Calinski-Harabasz指数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191016133200490.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
### 2.1.10 Davies-Bouldin指数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191016133256923.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
### 2.1.11 可能性矩阵

## 2.2 内部指标

考虑聚类结果的簇划分$C = {C_1, C_2,...,C_k}$,定义
$$\begin{aligned} \operatorname{avg}(C) &=\frac{2}{|C|(|C|-1)} \sum_{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \\ \operatorname{diam}(C) &=\max _{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \\ d_{\min }\left(C_{i}, C_{j}\right) &=\min _{\boldsymbol{x}_{i} \in C_{i}, \boldsymbol{x}_{j} \in C_{j}} \operatorname{dist}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \\ d_{\operatorname{cen}}\left(C_{i}, C_{j}\right) &=\operatorname{dist}\left(\boldsymbol{\mu}_{i}, \boldsymbol{\mu}_{j}\right) \end{aligned}$$

### 2.2.1 DB指数(Davies-Bouldin Index,简称DBI)
$\mathrm{DBI}=\frac{1}{k} \sum_{i=1}^{k} \max _{j \neq i}\left(\frac{\operatorname{avg}\left(C_{i}\right)+\operatorname{avg}\left(C_{j}\right)}{d_{\operatorname{cen}}\left(\boldsymbol{\mu}_{i}, \boldsymbol{\mu}_{j}\right)}\right)$
越小越好
### 2.2.2 Dunn指数(Dunn Index,简称DI)
$\mathrm{DI}=\min _{1 \leqslant i \leqslant k}\left\{\min _{j \neq i}\left(\frac{d_{\min }\left(C_{i}, C_{j}\right)}{\max _{1 \leqslant l} \operatorname{diam}\left(C_{l}\right)}\right)\right\}$
越大越好
### 2.2.3 统计量
1. $r^2$统计量：类间离差平方和之和在总离差平方和中所占的比例，该值越大说明聚类效果越好。该值总是随着聚类个数的减少而变小，故可以选择一个骤降点作为聚类个数的选择。
2. 半偏$r^2$统计量：是上一步$r^2$与这一步$r^2$值之差，故该值越大，说明上一次聚类效果越好。
3. 伪F统计量：越大说明这次聚类越好。
4. 伪$t^2$统计量：越大说明这次聚类越好。

>adjusted_mutual_info_score
>adjusted_rand_score
>completeness_score
>fowlkes_mallows_score
>homogeneity_score
>mutual_info_score
>normalized_mutual_info_score
>v_measure_score




# 4 其他指标
## 4.1 准确性指标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164532254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
## 4.2 单调性指标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164543509.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164735802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
