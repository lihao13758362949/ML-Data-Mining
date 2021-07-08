import time
import pandas as pd
import numpy as np
# import setting
# import EDA_Base
# import Preprocessing
# import baselib.feature_engineering

job = 'category'  # or regression
label = 'label'
EDA = True
auto_EDA = False

train_path = 'sample_submission.csv'
test_path = ''

# 0 setting
print('默认设置开始...')
default_time_start = time.time()

import setting

default_time_end = time.time()
print('默认设置完成，耗时： ', default_time_end - default_time_start, '秒')

# 1 EDA 输入：数据文件 输出：df_train,df_test
print('EDA开始...')
EDA_time_start = time.time()



print('读取数据开始...')
LoadData_time_start = time.time()
from EDA_Base import get_data

df_train = get_data(path=train_path, data_type='csv', header=0, names=None)
df_test = get_data(path=test_path, data_type='csv', header=0, names=None)
LoadData_time_end = time.time()
print('读取数据完成，耗时： ', LoadData_time_end - LoadData_time_start, '秒')

print('> df_train sample:', len(df_train))
print('>> positive-1 sample:', len(df_train[df_train[label] == 1]))
print('>> negtive-1 sample:', len(df_train[df_train[label] == 0]))
print('>> 0/1 sample:', len(df_train[df_train[label] == 0]) / len(df_train[df_train[label] == 1]))
print('> features number is:', len(df_train.columns))
print('> df_test sample:', len(df_test), '\n')

print('减少数据大小开始...')
from EDA_Base import reduce_mem_usage

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
print('减少数据大小完成')

if EDA:
    print('先看一下训练集和测试集分布是否类似（todo)')
    train_test_same = True

    if auto_EDA:
        from EDA_Base import one_key_EDA

        one_key_EDA(df_train)
    print('整体看一下训练集')
    from EDA_Base import easy_look

    is_dup, is_missing = easy_look(df_train)

    print('开始单变量分析')
    from EDA_Base import single_variable_EDA

    single_variable_EDA(df_train, label, job)
else:
    id_dup = False
    is_missing = False

EDA_time_end = time.time()
print('EDA完成，耗时： ', EDA_time_end - EDA_time_start, '秒')

# 2 数据预处理 输入：df_train,df_test 输出：all_data

print('训练集测试集分布一致化预处理（todo）')
if not train_test_same:
    pass

print('将训练集和测试集合并处理')
from Preprocessing import concat_train_and_test
all_data = concat_train_and_test(df_train, df_test)

if is_dup:
    print('重复值删除（todo）')
    # from sklearn.preprocessing import drop_duplicates这函数好像删了
    pass
if is_missing:
    print('缺失值处理（todo）')

    pass
print('偏态分布处理')
# np.log1p
# 3 特征工程 输入：all_data
# 输出：train_final.csv,test_final.csv或 train_for_tree.csv,test_for_tree.csv,train_for_lr.csv
from baselib.feature_engineering import *
# todo


# 4. model前准备 输入：train_final.csv,test_final.csv
df_train_final = pd.read_csv('train_final.csv')
df_test_final = pd.read_csv('test_final.csv')

features = [x for x in df_test_final.columns]

y_train = df_train_final[label]
X_train = df_train_final[features]
X_test = df_test_final[features]
## model_selection 输入：训练集X和y 输出：splits
## metrics 输出：metrics


# 5. model #输入：metrics, X_train, y_train, X_test 输出：scores,res,模型（feature_importances，best_iterations）
from model.lgb import lgb_model
feature_importances, metrics, best_iterations, predictions_lgb = lgb_model(X_train, y_train, X_test)
## model_params
