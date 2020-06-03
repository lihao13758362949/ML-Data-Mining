# 1 结构化竞赛
[百度-西交大数据竞赛2019Top4方案](https://mp.weixin.qq.com/s?srcid=&scene=23&sharer_sharetime=1569642063605&mid=2648042249&sharer_shareid=48deaea9fb8a9520544cf8fdae095a86&sn=e3b8291b4498f8dd698615ffd6599dc5&idx=1&__biz=MzA5NTI2NTY3Nw==&chksm=8860bdebbf1734fd20b05218a9fb1721ffe5e8c3ca4a18597911620a22e434f7bad258450c08&mpshare=1)
## 1.1 推荐
### 1.1.1 商品推荐
[安泰杯 —— 跨境电商智能算法大赛](https://tianchi.aliyun.com/competition/entrance/231718/forum)，预测用户购买商品top30。
### 1.1.1 销量预估
3.2.1 任务名称
Walmart Recruiting - Store Sales Forecasting
3.2.2 任务详情
Walmart 提供 2010-02-05 到 2012-11-01 期间的周销售记录作为训练数据，需要参赛选手建立模型预测 2012-11-02 到 2013-07-26 周销售量。比赛提供的特征数据包含：Store ID, Department ID, CPI，气温，汽油价格，失业率，是否节假日等。
3.2.3 获奖方案
● 1st place：Time series forecasting method: stlf + arima + ets。主要是基于时序序列的统计方法，大量使用了 Rob J Hyndman 的 forecast R 包。方案链接：https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125
● 2nd place：Time series forecasting + ML: arima + RF + LR + PCR。时序序列的统计方法+传统机器学习方法的混合；方案链接：https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023
● 16th place：Feature engineering + GBM。方案链接：https://github.com/ChenglongChen/Kaggle_Walmart-Recruiting-Store-Sales-Forecasting
3.2.4 常用工具
▲ R forecast package: https://cran.r-project.org/web/packages/forecast/index.html
▲ R GBM package: https://cran.r-project.org/web/packages/gbm/index.html
### 1.1.2 搜索相关性
3.3.1 任务名称
CrowdFlower Search Results Relevance
3.3.2 任务详情
比赛要求选手利用约几万个 (query, title, description) 元组的数据作为训练样本，构建模型预测其相关性打分 {1, 2, 3, 4}。比赛提供了 query, title和description的原始文本数据。比赛使用 Quadratic Weighted Kappa 作为评估标准，使得该任务有别于常见的回归和分类任务。
3.3.3 获奖方案
● 1st place：Data Cleaning + Feature Engineering + Base Model + Ensemble。对原始文本数据进行清洗后，提取了属性特征，距离特征和基于分组的统计特征等大量的特征，使用了不同的目标函数训练不同的模型（回归，分类，排序等），最后使用模型集成的方法对不同模型的预测结果进行融合。方案链接：https://github.com/ChenglongChen/Kaggle_CrowdFlower
● 2nd place：A Similar Workflow
● 3rd place： A Similar Workflow
3.3.4 常用工具
▲ NLTK: http://www.nltk.org/
▲ Gensim: https://radimrehurek.com/gensim/
▲ XGBoost: https://github.com/dmlc/xgboost
▲ RGF: https://github.com/baidu/fast_rgf
### 1.1.3 点击率预估
3.4.1 任务名称
Criteo Display Advertising Challenge
3.4.2 任务详情
经典的点击率预估比赛。该比赛中提供了7天的训练数据，1 天的测试数据。其中有13 个整数特征，26 个类别特征，均脱敏，因此无法知道具体特征含义。
3.4.3 获奖方案
● 1st place：GBDT 特征编码 + FFM。台大的队伍，借鉴了Facebook的方案 [6]，使用 GBDT 对特征进行编码，然后将编码后的特征以及其他特征输入到 Field-aware Factorization Machine（FFM） 中进行建模。方案链接：https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555
● 3rd place：Quadratic Feature Generation + FTRL。传统特征工程和 FTRL 线性模型的结合。方案链接：https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10534
● 4th place：Feature Engineering + Sparse DNN
3.4.4 常用工具
▲ Vowpal Wabbit: https://github.com/JohnLangford/vowpal_wabbit
▲ XGBoost: https://github.com/dmlc/xgboost
▲ LIBFFM: http://www.csie.ntu.edu.tw/~r01922136/libffm/



3.5.1 任务名称
Avazu Click-Through Rate Prediction
3.5.2 任务详情
点击率预估比赛。提供了 10 天的训练数据，1 天的测试数据，并且提供时间，banner 位置，site, app, device 特征等，8个脱敏类别特征。
3.5.3 获奖方案
● 1st place：Feature Engineering + FFM + Ensemble。还是台大的队伍，这次比赛，他们大量使用了 FFM，并只基于 FFM 进行集成。方案链接：https://www.kaggle.com/c/avazu-ctr-prediction/discussion/12608
● 2nd place：Feature Engineering + GBDT 特征编码 + FFM + Blending。Owenzhang（曾经长时间雄霸 Kaggle 排行榜第一）的竞赛方案。Owenzhang 的特征工程做得非常有参考价值。方案链接：https://github.com/owenzhang/kaggle-avazu
3.5.4 常用工具
▲ LIBFFM: http://www.csie.ntu.edu.tw/~r01922136/libffm/
▲ XGBoost: https://github.com/dmlc/xgboost

### 1.1.4 曝光量预测
1. 2019腾讯广告算法大赛
[TOP1](https://zhuanlan.zhihu.com/p/73062485)
[TOP5](https://mp.weixin.qq.com/s/j5YICHrkHLDm7OldPFPOjw)
