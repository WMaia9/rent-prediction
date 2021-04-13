import os
import settings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


#Ler os arquivos CSV
def read():
    train = pd.read_csv(os.path.join(settings.dir_processos, 'train.csv'))
    test = pd.read_csv(os.path.join(settings.dir_processos, 'test.csv'))
    print("Train set size:", train.shape)
    print("Test set size:", test.shape)
    print('START data processing', datetime.now(), )
    return train, test

def ajustes(train):
    train_ID = train['Id']

    # Now drop the  'Id' colum since it's unnecessary for  the prediction process.
    train.drop(['Id'], axis=1, inplace=True)

    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    train["Price"] = np.log1p(train["Price"])
    y = train.Price.reset_index(drop=True)
    features = train.drop(['Price'], axis=1)

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)

    features.update(features[objects].fillna('None'))

    # simplified features
    print(features.shape)
    final_features = pd.get_dummies(features).reset_index(drop=True)
    print(final_features.shape)

    X = final_features.iloc[:len(y), :]
    X_sub = final_features.iloc[len(X):, :]

    print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)
    return X, y

def models(X, y):
    print('START ML', datetime.now(), )

    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

    # rmsle
    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    # build our model scoring function
    def cv_rmse(model, X=X):
        rmse = np.sqrt(-cross_val_score(model, X, y,
                                        scoring="neg_mean_squared_error",
                                        cv=kfolds))
        return (rmse)

    # setup models
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

    ridge = make_pipeline(RobustScaler(),
                          RidgeCV(alphas=alphas_alt, cv=kfolds))

    lasso = make_pipeline(RobustScaler(),
                          LassoCV(max_iter=1e7, alphas=alphas2,
                                  random_state=42, cv=kfolds))

    elasticnet = make_pipeline(RobustScaler(),
                               ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                            cv=kfolds, l1_ratio=e_l1ratio))

    svr = make_pipeline(RobustScaler(),
                        SVR(C=20, epsilon=0.008, gamma=0.0003, ))

    gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)

    lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200,
                             bagging_seed=7,
                             feature_fraction_seed=7, verbose=-1)

    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7,
                           colsample_bytree=0.7, nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006)

    # stack
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                    meta_regressor=xgboost, use_features_in_secondary=True)

    score = cv_rmse(ridge)
    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(lasso)
    print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(elasticnet)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(svr)
    print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(lightgbm)
    print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(gbr)
    print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    score = cv_rmse(xgboost)
    print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    print('START Fit')
    print(datetime.now(), 'StackingCVRegressor')
    stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
    print(datetime.now(), 'elasticnet')
    elastic_model_full_data = elasticnet.fit(X, y)
    print(datetime.now(), 'lasso')
    lasso_model_full_data = lasso.fit(X, y)
    print(datetime.now(), 'ridge')
    ridge_model_full_data = ridge.fit(X, y)
    print(datetime.now(), 'svr')
    svr_model_full_data = svr.fit(X, y)
    print(datetime.now(), 'GradientBoosting')
    gbr_model_full_data = gbr.fit(X, y)
    print(datetime.now(), 'xgboost')
    xgb_model_full_data = xgboost.fit(X, y)
    print(datetime.now(), 'lightgbm')
    lgb_model_full_data = lightgbm.fit(X, y)

    #Função para juntar todos os modelos
    def blend_models_predict(X):
        return ((0.1 * elastic_model_full_data.predict(X)) + \
                (0.1 * lasso_model_full_data.predict(X)) + \
                (0.1 * ridge_model_full_data.predict(X)) + \
                (0.1 * svr_model_full_data.predict(X)) + \
                (0.1 * gbr_model_full_data.predict(X)) + \
                (0.15 * xgb_model_full_data.predict(X)) + \
                (0.1 * lgb_model_full_data.predict(X)) + \
                (0.25 * stack_gen_model.predict(np.array(X))))


    return rmsle(y, blend_models_predict(X))

if __name__ == "__main__":
    train, test = read()
    X_train, y_train = ajustes(train)
    print('TRAIN score on CV')
    rmsle_train = models(X_train, y_train)
    '''X_test, y_test = ajustes(test)
    print('TEST score on CV')
    rmsle_test = models(X_test, y_test)'''

    print('RMSLE score on train data:', rmsle_train)
    #print('RMSLE score on test data:', rmsle_test)


