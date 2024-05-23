#!/usr/bin/env python
# coding=utf-8

r"""
利用xgboost包来进行 gbdt 模型的学习。

https://github.com/dmlc/xgboost

"""

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import metrics
from tqdm import tqdm_notebook
tqdm_notebook().pandas()


def cal_metrics(model, x_train, y_train, x_test, y_test):
    """ Calculate AUC and accuracy metric
        Args:
            model: model that need to be evaluated
            x_train: feature of training set
            y_train: target of training set
            x_test: feature of test set
            y_test: target of test set
    """
    y_train_pred_label = model.predict(x_train)
    y_train_pred_proba = model.predict_proba(x_train)
    accuracy = accuracy_score(y_train, y_train_pred_label)
    auc = roc_auc_score(y_train, y_train_pred_proba[:, 1])
    print("Train set: accuracy: %.2f%%" % (accuracy*100.0))
    print("Train set: auc: %.2f%%" % (auc*100.0))
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_test_proba = model.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_test_proba[:, 1])
    print("Test set: accuracy: %.2f%%" % (accuracy*100.0))
    print("Test set: auc: %.2f%%" % (auc*100.0))


def model_iteration_analysis(alg, feature, predictors, use_train_cv=True,
                             cv_folds=5, early_stopping_rounds=50):
    """ The optimal iteration times of the model are analyzed
        Args:
            alg: model
            feature: feature of train set
            predictors: target of train set
            use_train_cv: whether to cross-validate
            cv_folds: the training set id divided into several parts
            early_stopping_rounds: observation window size of iteration number
        Return:
            alg: optimal model
    """
    if use_train_cv:
        xgb_param = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(feature, label=predictors)
        # 'cv' function， can use cross validation on each iteration and return the desired number of decision trees.
        cv_result = xgb.cv(xgb_param, xgb_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        print("cv_result---", cv_result.shape[0])
        print(cv_result)
        alg.set_params(n_estimators=cv_result.shape[0])

    # Fit the algorithm on the data
    alg.fit(feature, predictors, eval_metric='auc')
    # Predict training set:
    predictions = alg.predict(feature)
    pred_prob = alg.predict_proba(feature)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(predictors, predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(predictors, pred_prob))
    return alg


if __name__ == "__main__":
    # build Xgboost model
    import warnings
    warnings.filterwarnings('ignore')
    TEST_RATIO = 0.3
    RANDOM_STATE = 33
    xgb_X = full_preprocessing_feature[cols]
    xgb_Y = full_feature['target']
    X_full_train, X_full_test, y_full_train, y_full_test = \
        train_test_split(xgb_X, xgb_Y, test_size=TEST_RATIO, random_state=RANDOM_STATE)

    # build the base model using the initial values
    base_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        booster='gbtree',
        colsample_bytree=1,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=1,
        verbosity=0,
        objective='binary:logistic',
        seed=666
    )

    # train model
    base_model.fit(X_full_train, y_full_train)
    cal_metrics(base_model, X_full_train, y_full_train, X_full_test, y_full_test)

    # adjust tree structure
    param_tree_struction = {
        'max_depth': range(3, 16, 2),
        'min_child_weight': range(1, 8, 2)
    }
    # grid search
    full_tree_struction_gsearch = GridSearchCV(estimator=base_model,
                                               param_grid=param_tree_struction, scoring='roc_auc',
                                               cv=5, verbose=0, iid=False)
    full_tree_struction_gsearch.fit(X_full_train, y_full_train)
    print(full_tree_struction_gsearch.best_params_, full_tree_struction_gsearch.best_score_,
          full_tree_struction_gsearch.best_estimator_)
    cal_metrics(full_tree_struction_gsearch.best_estimator_, X_full_train, y_full_train, X_full_test, y_full_test)

    # continue to adjust the tree structure more precisely
    param_tree_struction2 = {
        'max_depth': [6, 7, 8],
        'min_child_weight': [4, 5, 6]
    }

    tree_struction_gsearch2 = GridSearchCV(estimator=base_model,param_grid=param_tree_struction2,
                                           scoring='roc_auc', cv=5, verbose=0, iid=False)
    tree_struction_gsearch2.fit(X_full_train, y_full_train)
    print(tree_struction_gsearch2.best_params_, tree_struction_gsearch2.best_score_, tree_struction_gsearch2.best_estimator_)
    cal_metrics(tree_struction_gsearch2.best_estimator_, X_full_train, y_full_train, X_full_test, y_full_test)

    adjust_tree_struction_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        booster='gbtree',
        colsample_bytree=1,
        gamma=0,
        max_depth=6,
        min_child_weight=6,
        n_jobs=4,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=1,
        verbosity=0,
        objective='binary:logistic',
        seed=666
    )

    # adjusting the Gamma parameter
    param_gamma = {
        'gamma': [i / 10 for i in range(0, 10)]
    }
    gamma_gsearch = GridSearchCV(estimator=adjust_tree_struction_model, param_grid=param_gamma, scoring='roc_auc',
                                 cv=5, verbose=0, iid=False)
    gamma_gsearch.fit(X_full_train, y_full_train)
    print(gamma_gsearch.best_params_, gamma_gsearch.best_score_, gamma_gsearch.best_estimator_)
    # calculate AUC and accuracy metric
    cal_metrics(gamma_gsearch.best_estimator_, X_full_train, y_full_train, X_full_test, y_full_test)


    adjust_gamma_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        booster='gbtree',
        colsample_bytree=1,
        gamma=0,
        max_depth=6,
        min_child_weight=6,
        n_jobs=4,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=1,
        verbosity=0,
        objective='binary:logistic',
        seed=666
    )
    # adjust sample ratio and column sampling ratio parameters
    param_sample = {
        'subsample': [i / 10 for i in range(6, 11)],
        'colsample_bytree': [i / 10 for i in range(6, 11)],
    }
    sample_gsearch = GridSearchCV(estimator=adjust_gamma_model, param_grid=param_sample,
                                  scoring='roc_auc', cv=5, verbose=0, iid=False)
    sample_gsearch.fit(X_full_train, y_full_train)
    print(sample_gsearch.best_params_, sample_gsearch.best_score_, sample_gsearch.best_estimator_)
    cal_metrics(sample_gsearch.best_estimator_, X_full_train, y_full_train, X_full_test, y_full_test)

    adjust_sample_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        booster='gbtree',
        colsample_bytree=0.8,
        gamma=0,
        max_depth=6,
        min_child_weight=6,
        n_jobs=4,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=0.8,
        verbosity=0,
        objective='binary:logistic',
        seed=666
    )

    # adjust regularization param
    param_L = {
        'reg_lambda': [1e-5, 1e-3, 1e-2, 1e-1, 1, 100],
    }
    L_gsearch = GridSearchCV(estimator=adjust_sample_model, param_grid=param_L, scoring='roc_auc', cv=5, verbose=0, iid=False)
    L_gsearch.fit(X_full_train, y_full_train)
    print(L_gsearch.best_params_, L_gsearch.best_score_, L_gsearch.best_estimator_)
    model_iteration_analysis(L_gsearch.best_estimator_, X_full_train, y_full_train, early_stopping_rounds=30)

    # adjusted learning rate
    param_learning_rate = {
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
    }
    learning_rate_gsearch = GridSearchCV(estimator=adjust_sample_model, param_grid=param_learning_rate, scoring='roc_auc', cv=5, verbose=0, iid=False)
    learning_rate_gsearch.fit(X_full_train, y_full_train)
    print(learning_rate_gsearch.best_params_, learning_rate_gsearch.best_score_, learning_rate_gsearch.best_estimator_)

    model_iteration_analysis(learning_rate_gsearch.best_estimator_,
                             X_full_train, y_full_train, early_stopping_rounds=30)

    # optimal model
    best_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        booster='gbtree',
        colsample_bytree=0.7,
        gamma=0.6,
        max_depth=6,
        min_child_weight=2,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=0.9,
        verbosity=0,
        objective='binary:logistic',
        seed=666
    )
    best_model.fit(X_full_train, y_full_train)
    print('--- the training set and test set metrics of Xgboost model ---\n')
    cal_metrics(best_model, X_full_train,y_full_train, X_full_test, y_full_test)
    print('\n')
    print(best_model.get_xgb_params())

    # according to XgBoost model, the importance of features was analyzed
    print('\n')
    print('---according to xGBoost model, the importance of features was analyzed---\n')
    from xgboost import plot_importance
    fig, ax = plt.subplots(figsize=(10, 15))
    plot_importance(best_model, height=0.5, max_num_features=100, ax=ax)
    plt.show()

    # Draw ROC curve
    y_pred_proba = best_model.predict_proba(X_full_test)
    fpr, tpr, thresholds = roc_curve(y_full_test, y_pred_proba[:, 1])
    print('---ROC curve of xgboost model ---\n')
    plt.title('roc_curve of xgboost (AUC=%.4f)' % (roc_auc_score(y_full_test, y_pred_proba[:, 1])))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    plt.show()
