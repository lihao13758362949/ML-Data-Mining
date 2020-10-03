# 分类器评价指标
1. **准确率/精度**：正确预测的样本数/样本数
2. **平衡准确率**：每个类上获得的召回率的平均值
3. average_precision_score
4. **（查准率）精确率**：$\displaystyle precision=\frac{tp}{tp+fp}$，直观地说是分类器不将负样本标记为正样本的能力。
5. **（查全率）召回率**：$\displaystyle recall=\frac{tp}{tp+fn}$，直观上是分类器发现所有正样本的能力。
6. **F1**：$F1 = \displaystyle\frac{2 (precision \times recall)} {precision + recall}$，实际上是精确率和召回率的调和平均。
7. $F_\beta$：$F_\beta=\displaystyle\frac{(1+\beta^2)\times precision \times recall} {\beta^2\times precision + recall}$，加权调和平均，考虑了偏好，$\beta>1$时查全率更偏好。
8. Micro-F1：多类别问题中的F1指标，计算出所有类别总的Precision和Recall，然后计算F1。
9. Macro-F1：多类别问题中的F1指标，计算出每一个类的Precison和Recall后计算F1，最后将F1平均。
10. **logloss**：也叫做**逻辑损失**或**交叉熵损失**。$\displaystyle logloss=-\frac{1}{m}\sum^m_{i=1}(y_ilog(p_i)*\delta+(1-y_i)log(1-p_i))$或$logloss=-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))$
>brier_score_loss

>jaccard_score
11. **roc_auc_score**：将测试样本排序，越可能是正例的排在越前面，定义两个概念：真正例率$TPR=\frac{TP}{TP+FN}$，假正例率$FPR=\frac{FP}{TN+FP}$，和P-R曲线类似操作（按顺序逐个把样本作为正例进行评价）则得到roc曲线。（本图来源《机器学习》周志华）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190926125950662.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)

AUC为ROC曲线以下的面积，可以看出AUC越大，模型的效果越好。 

# 
