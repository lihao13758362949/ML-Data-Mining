# 分类器评价指标
7. $F_\beta$：$F_\beta=\displaystyle\frac{(1+\beta^2)\times precision \times recall} {\beta^2\times precision + recall}$，加权调和平均，考虑了偏好，$\beta>1$时查全率更偏好。
8. Micro-F1：多类别问题中的F1指标，计算出所有类别总的Precision和Recall，然后计算F1。
9. Macro-F1：多类别问题中的F1指标，计算出每一个类的Precison和Recall后计算F1，最后将F1平均。







# 4 其他指标
## 4.1 准确性指标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164532254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
## 4.2 单调性指标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164543509.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422164735802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
