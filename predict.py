import os
import settings
from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold, cross_val_score
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, Ridge
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data CSV
def read():
    train = pd.read_csv(os.path.join(settings.dir_processos, 'train.csv'))
    test = pd.read_csv(os.path.join(settings.dir_processos, 'test.csv'))
    print("Train set size:", train.shape)
    print("Test set size:", test.shape)
    print('START data processing', datetime.now(), )


    return train, test

# Feature Engineering
def features(train, test):
    train.drop('Id', inplace=True, axis=1)  # Drop column and update same dataframe
    test.drop('Id', inplace=True, axis=1)
    print(f'After dropping Id feature, shape of Train Data: {train.shape}, Test Data: {test.shape}')

    y_train = train.Price
    # Drops the current index of the DataFrame and replaces it with an index of increasing integers
    all_data = pd.concat((train, test)).reset_index(drop=True)
    # Delete SalePrice from all data
    all_data.drop(['Price'], axis=1, inplace=True)

    cat_feats_nominal = ['Type', 'Region']
    cat_feats_nominal_one_hot = pd.get_dummies(all_data[cat_feats_nominal], drop_first=True).reset_index(drop=True)

    # First we need to drop the catgorical nominal columns from all_data
    all_data = all_data.drop(cat_feats_nominal, axis='columns')

    all_data = pd.concat([all_data, cat_feats_nominal_one_hot], axis='columns')

    train = all_data[:len(y_train)]
    test = all_data[len(y_train):]
    print(f'Shape of train: {train.shape}, test:{test.shape} after Features Engineering')

    y_train = np.log1p(y_train)

    return y_train, train, test

# Evaluation Metric
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, kf):
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


def predict(y_train, train, test):
    y_test = pd.read_csv(os.path.join(settings.dir_processos, 'test.csv'))
    y_test = y_test['Price']
    y_test = np.log1p(y_test)

    def score_model(model_reg):
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        score_model_reg = cv_rmse(model_reg, kf)
        print(f'Score Model => Mean: {score_model_reg.mean()}, Std: {score_model_reg.std()}')
        cv_scores.append(score_model_reg.mean())
        cv_std.append(score_model_reg.std())

    n_folds = 5
    random_state = 42
    cv_scores = []
    cv_std = []
    random_state = 42

    print('\nLinear Regression:')
    linear_reg = LinearRegression()
    score_model(linear_reg)
    linear_reg.fit(train.values, y_train)
    y_pred = linear_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nBayesian Ridge:')
    bayesian_ridge_reg = BayesianRidge(alpha_1=2.104047761709729e-05,
                                       alpha_2=8.87111148542247e-06,
                                       lambda_1=0.9517616852006183,
                                       lambda_2=0.016369928482509982,
                                       compute_score=False)
    score_model(bayesian_ridge_reg)
    bayesian_ridge_reg.fit(train.values, y_train)
    y_pred = bayesian_ridge_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nLasso:')
    lasso_reg = pipeline.Pipeline([("scaling", preprocessing.RobustScaler()),
                                   ("lasso", Lasso(alpha=0.0004225349823414949,
                                                   max_iter=1000000,
                                                   tol=0.001,
                                                   random_state=random_state))])
    score_model(lasso_reg)
    lasso_reg.fit(train.values, y_train)
    y_pred = lasso_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nElastic Net:')
    elastic_net_reg = pipeline.Pipeline([("scaling", preprocessing.RobustScaler()),
                                         ("elastic_net", ElasticNet(alpha=0.0005033042674715873,
                                                                    l1_ratio=0.8201479505715717,
                                                                    positive=True,
                                                                    precompute=False,
                                                                    selection='random',
                                                                    max_iter=10000000,
                                                                    tol=0.001,
                                                                    random_state=random_state))])

    score_model(elastic_net_reg)
    elastic_net_reg.fit(train.values, y_train)
    y_pred = elastic_net_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nRidge Regression:')
    ridge_reg = pipeline.Pipeline([("scaling", preprocessing.RobustScaler()),
                                   ("ridge", Ridge(alpha=12.773681311355642,
                                                   random_state=random_state))])
    score_model(ridge_reg)
    ridge_reg.fit(train.values, y_train)
    y_pred = ridge_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nSupport Vector Regression:')
    svr_reg = pipeline.Pipeline([("scaling", preprocessing.RobustScaler()),
                                 ("svr", svm.SVR(C=46,
                                                 epsilon=0.009019504329938493,
                                                 gamma=0.0003434802243340735))])

    score_model(svr_reg)
    svr_reg.fit(train.values, y_train)
    y_pred = svr_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nGradient Boosting Regressor:')
    gbr_reg = GradientBoostingRegressor(n_estimators=2501,
                                        learning_rate=0.03221041191991256,
                                        random_state=random_state)
    score_model(gbr_reg)
    gbr_reg.fit(train.values, y_train)
    y_pred = gbr_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nLGBM Regressor:')
    lgbm_reg = LGBMRegressor(objective='regression',
                             num_leaves=4,
                             learning_rate=0.01,
                             n_estimators=5000,
                             max_bin=200,
                             bagging_seed=7,
                             feature_fraction_seed=7,
                             verbose=-1)
    score_model(lgbm_reg)
    lgbm_reg.fit(train.values, y_train)
    y_pred = lgbm_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nXGB Regressor:')
    xgb_reg = XGBRegressor(learning_rate=0.00922801668420645,
                           n_estimators=4492,
                           max_depth=4,
                           min_child_weight=0.019476741626353912,
                           gamma=0.0038933017613795614,
                           subsample=0.3075828286669299,
                           colsample_bytree=0.16053941121623433,
                           scale_pos_weight=3,
                           reg_alpha=6.89051576939588e-05,
                           objective='reg:squarederror',
                           random_state=random_state)

    score_model(xgb_reg)
    xgb_reg.fit(train.values, y_train)
    y_pred = xgb_reg.predict(test.values)
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    print('\nStacking:')
    estimators = (linear_reg, svr_reg, bayesian_ridge_reg, ridge_reg, lasso_reg,
                  elastic_net_reg, gbr_reg, lgbm_reg, xgb_reg)
    final_estimator = xgb_reg
    stacking_cv_reg = StackingCVRegressor(regressors=estimators,
                                          meta_regressor=final_estimator,
                                          use_features_in_secondary=True,
                                          random_state=random_state)

    kf = KFold(n_splits=n_folds,
               random_state=random_state,
               shuffle=True)
    score_model_reg = np.sqrt(
        -cross_val_score(stacking_cv_reg, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    print(f'score_model_reg => mean: {score_model_reg.mean()}, std: {score_model_reg.std()}')
    cv_scores.append(score_model_reg.mean())
    cv_std.append(score_model_reg.std())

    models_with_weights = {linear_reg: 0.005,
                           svr_reg: 0.005,
                           bayesian_ridge_reg: 0.005,
                           ridge_reg: 0.05,
                           lasso_reg: 0.1,
                           elastic_net_reg: 0.1,
                           gbr_reg: 0.1,
                           lgbm_reg: 0.1,
                           xgb_reg: 0.1,
                           stacking_cv_reg: 0.435}

    print('\nBlended:')
    # Contains predicted values for Price
    blended_train_pred = pd.DataFrame()
    blended_test_pred = pd.DataFrame()
    for model, weight in models_with_weights.items():
        # print(f"Model: {str(model)}, Weight: {weight}")    
        if re.search('StackingCVRegressor', str(model), re.I):
            # For stacking_cv model we will pass 'train.values' and 'test.values', 
            # To avoid error : ValueError: feature_names mismatch:
            model.fit(train.values, y_train)
            blended_train_pred[model] = weight * model.predict(train.values)
            blended_test_pred[model] = weight * model.predict(test.values)
        else:
            model.fit(train, y_train)
            blended_train_pred[model] = weight * model.predict(train)
            blended_test_pred[model] = weight * model.predict(test)

    print(f'blended_train_pred.shape: {blended_train_pred.shape}')
    print(f'blended_test_pred.shape: {blended_test_pred.shape}')

    # Find score using full training data
    train_score = rmse(y_train, blended_train_pred.sum(axis='columns'))
    print(f'RMSE on Train Data: {train_score}')
    y_pred = blended_test_pred.sum(axis='columns')
    print('RMSE on Test Data:', rmse(y_test, y_pred))

    # Predict on test data
    test_pred = np.floor(np.expm1(blended_test_pred.sum(axis='columns')))

    # Saving parameters
    #filename = 'finalized_model.sav'
    #stacking_cv_reg.dump(linear_reg, filename)
    
    return test_pred

if __name__ == "__main__":
    train, test = read()
    test_sub = test[['Id', 'Price']]
    print('\nSTART data processing', datetime.now(), )
    y_train, train, test = features(train, test)
    print('\nSTART ML', datetime.now(), )
    test_pred = predict(y_train, train, test)
    print('\nWriting Data', datetime.now(), )
    test_sub.iloc[:, 1] = test_pred
    test_sub.to_csv(os.path.join(settings.dir_processos, "test_pred.csv"), index=False)
