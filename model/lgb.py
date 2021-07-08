"""
lgb.py
LightGBM
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 50 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(20, 12))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:50]))),
            df['importance_normalized'].head(50),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:50]))))
    ax.set_yticklabels(df['feature'].head(50))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df


def rmse(labels, preds):
    # train 中的feval自定义评价函数举例
    preds = preds.get_label()
    loss = np.sqrt(mean_squared_error(labels, preds))
    return 'rmsle', loss, False


def get_lgb_params():
    lgb_params = {'boosting_type': 'gbdt',
                  'num_leaves': 20,
                  'num_iterations': 500,
                  'min_data_in_leaf': 20,
                  'objective': 'regression',
                  'max_depth': 6,
                  'learning_rate': 0.01,
                  "feature_fraction": 0.8,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.8,
                  "bagging_seed": 11,
                  "metric": 'rmse',
                  "lambda_l1": 0.1,
                  "verbosity": -1,
                  'random_state': 927
                  }
    return lgb_params


def lgb_model(X_train, y_train, X_test, exclude_columns=None):
    print(f'features count: {len(X_train.columns)}')
    print(f'features: {list(X_train.columns)}')
    features = [f for f in X_train.columns if f not in exclude_columns]

    params = get_lgb_params()
    folds = KFold(n_splits=5, shuffle=True, random_state=2021)
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros(len(X_test))
    feature_importance_values = []
    valid_scores = []
    train_scores = []
    best_iterations = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], y_train.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                        early_stopping_rounds=100)

        feature_importance_values.append(clf.feature_importance())
        best_iteration = clf.best_iteration
        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        print(clf.best_score)
        valid_score = clf.best_score['valid_1']['rmse']
        train_score = clf.best_score['training']['rmse']
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        best_iterations.append(best_iteration)

        gc.enable()
        del clf, trn_data, val_data
        gc.collect()
    # oof_lgb_final = np.argmax(oof_lgb, axis=1)
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train, squared=False)))
    # pred_label = np.argmax(prediction, axis=1)
    valid_scores_softmax = np.exp(np.array(valid_scores)) / np.sum(np.exp(np.array(valid_scores)))
    feature_importance_values = np.array(feature_importance_values).T

    feature_importances = pd.DataFrame({'feature': features,
                                        'importance': feature_importance_values.dot(valid_scores_softmax)})

    fold_names = list(range(folds.get_n_splits()))
    fold_names.append('overall')

    valid_loss = np.sqrt(mean_squared_error(y_train, oof_lgb))
    valid_scores.append(valid_loss)
    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return feature_importances, metrics, best_iterations, predictions_lgb


def final_submit(predictions_lgb):
    submission = pd.DataFrame({'id': np.arange(600), 'revenue': predictions_lgb})
