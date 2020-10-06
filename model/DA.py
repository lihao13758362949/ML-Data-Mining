from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

LDA(solver=’svd’, shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

#solver：svd：奇异值分解，是默认的求解器，不计算协方差矩阵，因此建议用于具有大量特征的数据。lsqr：最小二乘解。eigen：特征值分解。
#shrinkage：收缩率，可以在训练样本数量比特征数量少的情况下改进协方差矩阵的估计。可以设置为auto或[0,1]的数。指定为auto时需要将sover设置成lsqr或eigen。
#priors
#n_components：类别数
#store_covariance
#tol：SVD求解中用于秩和估计的阈值。









from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

QDA(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
