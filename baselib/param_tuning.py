# 参数调优
# 基于optuna
import optuna


def objective(trial):
    print(f'******************* 调参开始喽(●• ̀ω•́ )✧ *******************')
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 32),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 30),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 50),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.001, 10),
        "random_state": 927
    }

    out_of_fold = np.zeros(X.shape[0])
    kf = KFold(n_splits=5, random_state=927)
    for i, (tr, va) in enumerate(kf.split(X)):
        # print(f'******************* 第{i+1}次CV开始喽(●• ̀ω•́ )✧ *******************')
        tr_x, tr_y = X.iloc[tr], y_log.iloc[tr]
        va_x, va_y = X.iloc[va], y_log.iloc[va]
        lgb_tr = lgb.Dataset(tr_x, tr_y)
        lgb_va = lgb.Dataset(va_x, va_y, reference=lgb_tr)

        model = lgb.train(params=params, train_set=lgb_tr, valid_sets=[lgb_tr, lgb_va],
                          valid_names=['train', 'valid'], num_boost_round=10000,
                          categorical_feature=cat_cols,
                          feval=rmsle,
                          early_stopping_rounds=100, verbose_eval=0)

        best_iteration = model.best_iteration
        out_of_fold[va] = model.predict(X.iloc[va], num_iteration=best_iteration)
        valid_loss = np.sqrt(mean_squared_error(y_log, out_of_fold))
    print(f'rmsle: {valid_loss}')
    return valid_loss


study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=27), direction="minimize",
    study_name='lgb_1117'
)

study.optimize(objective, n_trials=70)

print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("     {}: {}".format(key, value))