import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv('train_x.csv')
Y_train = pd.read_csv('train_y.csv')
y_train = Y_train.values.ravel()
RF = RandomForestClassifier(n_estimators = 300, random_state = 42)
RF.fit(X_train, y_train)

X_test = pd.read_csv('test_x.csv')
y_test = RF.predict(X_test)
y_test2 = y_test.reshape(y_test.shape[0],1)
y_test2_pd = pd.DataFrame(y_test2)
y_test2_pd.to_csv('test_y.csv',header=False)

X_pred = pd.read_csv('pred_x.csv')
y_pred = RF.predict(X_test)
y_pred2 = y_pred.reshape(y_pred.shape[0],1)
y_pred2_pd = pd.DataFrame(y_pred2)
y_pred2_pd.to_csv('pred_y.csv',header=False)
