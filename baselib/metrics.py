
def metric(y_true,y_pred,type=reg):
    ## type：指标类型：reg回归指标，cls：分类指标，cluster:聚类指标
    ## y_true：真实值，y_pred：预测值
    
    from sklearn.metrics import *
  
    if type==reg:
        # 回归器评价指标
        
        mean_squared_error(y_true, y_pred, sample_weight=None, multioutput=’uniform_average’)# 均方误差
        max_error(y_true, y_pred)# 最大误差
        mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput=’uniform_average’)# 平均绝对误差
        #均方根误差
        # >>explained_variance_score
        # >mean_squared_log_error
        # >median_absolute_error
        # >r2_score
        # >Mean Absolute Percent Error (MAPE)
    if type==cls:
        # 分类器评价指标
        accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)# 准确率
        # normalize：默认返回正确率，若为False则返回预测正确的样本数。

        balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)#平衡准确率：每个类上获得的召回率的平均值
        # adjusted：

        average_precision_score(y_true, y_score, average=’macro’, pos_label=1, sample_weight=None)
        recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#（查全率）召回率tp/（tp+fn），直观上是分类器发现所有正样本的能力。
        precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#（查准率）精确率tp/（tp+fp），直观地说是分类器不将负样本标记为正样本的能力。
        f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#F1实际上是精确率和召回率的调和平均。
        # average：可指定micro和macro
        
        #macro-recall/p/f1是先计算再平均，而micro-recall/p/f1是先平均再计算
        
        log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)#也叫做逻辑损失或交叉熵损失
        # average_precision_score
        # brier_score_loss
        # jaccard_score

        roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None, max_fpr=None)#roc曲线的auc分数
        classification_report(y_test, credit_pred)#综合报告
        confusion_matrix(y_test, credit_pred)# 混淆矩阵
    if type=cluster:
        # 聚类器评价指标
        #聚类的好坏不存在绝对标准
        adjusted_rand_score(labels_true, labels_pred)：ARI指数
        mutual_info_score(labels_true, labels_pred, contingency=None)：互信息
        adjusted_mutual_info_score(labels_true, labels_pred, average_method=’warn’)
        normalized_mutual_info_score(labels_true, labels_pred, average_method=’warn’)
        completeness_score(labels_true, labels_pred)完备性
        homogeneity_score(labels_true, labels_pred)：同质性
        homogeneity_completeness_v_measure(labels_true, labels_pred, beta=1.0)
        v_measure_score(labels_true, labels_pred, beta=1.0)
        fowlkes_mallows_score(labels_true, labels_pred, sparse=False)
        silhouette_score(X, labels, metric=’euclidean’, sample_size=None, random_state=None, **kwds) 轮廓系数
        calinski_harabasz_score(X, labels)
        davies_bouldin_score(X, labels)
        contingency_matrix(labels_true, labels_pred, eps=None, sparse=False)

