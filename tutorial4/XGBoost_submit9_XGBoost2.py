import pandas as pd
import numpy as np
import xgboost as xgb 
from xgboost import XGBClassifier 

X_train = pd.read_csv('train_x.csv')
Y_train = pd.read_csv('train_y.csv')

X_train = X_train.values
y_train = Y_train.values.ravel()
final_params = {
    'n_estimators':1000, 
    'max_depth':50,
    'min_child_weight':10,
    'gamma':0.1,
    'colsample_bytree':.5,
    'subsample':.8,
    'reg_alpha':0,
    'reg_lambda':1,
    'learning_rate':.1
}
XGB = XGBClassifier(**final_params, seed_size=42)
XGB.fit(X_train, y_train)

y_eval = XGB.predict(X_train)
y_eval_pd=pd.DataFrame(y_eval)
y_eval_pd.to_csv('eval_y.csv',header=None)

X_test = pd.read_csv('test_x.csv')
X_test = X_test.values
y_test = XGB.predict(X_test)
y_test2 = y_test.reshape(y_test.shape[0],1)
y_test2_pd = pd.DataFrame(y_test2)
y_test2_pd.to_csv('test_y.csv',header=False)

X_train = pd.read_csv('train_all.csv',header=None)
X_train = X_train.values
y_test = XGB.predict(X_train)
y_test2 = y_test.reshape(y_test.shape[0],1)
y_test2_pd = pd.DataFrame(y_test2)
y_test2_pd.to_csv('train_all_y.csv',header=None)



X_pred = pd.read_csv('pred_x.csv',header=None)
X_pred = X_pred.values
y_pred = XGB.predict_proba(X_pred)
y_pred2 = y_pred.reshape(y_pred.shape[0],2)
y_pred2_pd = pd.DataFrame(y_pred2[:,1])
y_pred2_pd.to_csv('pred_y.csv',header=False)

