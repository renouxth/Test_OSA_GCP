# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:58:45 2016

@author: RENOUXTH
"""

import  os

os.environ['PATH'] = os.environ['PATH'] + ';C:\\git-sdk-64\\mingw64\\bin'

os.environ['PATH'] = os.environ['PATH'] + ';C:\\git-sdk-64\\xgboost\\lib'

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import operator
import matplotlib.pyplot as plt

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def rmspe(y, yhat):
    yhat[yhat<0]=0
    return np.nanmean(np.absolute(y-yhat))
    
#    np.minimum(np.absolute(y-yhat)/yhat, 1)
    
def rmspe_xg(yhat, y):
    y = y.get_label()
    #y = y.get_label()
    yhat = yhat
    #yhat = yhat
    return "rmspe", rmspe(y,yhat)
    
# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
     # Use some properties directly
    features.extend(data.columns)
    
    return data
    
## Start of main script

print("Load the training, test and store data using pandas")
##types = {'Annee' : np.dtype(int),
       ##  'Type_Offre': np.dtype(int),
       ##  'Code_Produit': np.dtype(int),
       ##  'Previ_Volume' : np.dtype(float),
       ##  'Nb_Jours': np.dtype(float),
       ##  'S_debut': np.dtype(float),
       ##  'UB': np.dtype(int),
       ##  'Taux_Degradation' : np.dtype(int),
       ##  'Magasin' : np.dtype(int),
       ##  'VMH': np.dtype(float),
       ##  'Baseline_Magasins' : np.dtype(float)}
        
##df = pd.read_csv('/Anaconda3/BDD Promo.csv', dtype=types, sep=';')
df = pd.read_csv('/Anaconda3/BDD Promo.csv', sep=';')

# Get rid of unnecessary features

#df = df.drop(['Jour', 'EAN_13', 'Libelle'], axis=1)

print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
df = df[df["Ventes"] > 0]

#df['Ventes'] = np.log1p(df['Ventes'])

#df['Moyenne'] = df['Moyenne'] 
#df['Ventes_S-1'] = df['Ventes_S-1'] 
#df['Ventes_S-2'] = df['Ventes_S-2'] 
#df['Ventes_S-3'] = df['Ventes_S-3'] 
#df['Moyenne'] = np.log1p(df['Moyenne'])
#df['Ventes_S-1'] = np.log1p(df['Ventes_S-1'])
#df['Ventes_S-2'] = np.log1p(df['Ventes_S-2'])
#df['Ventes_S-3'] = np.log1p(df['Ventes_S-3'])
#df['Poids_des_Jours'] = np.log1p(df['Poids_des_Jours'])

# Preparing Train, Test and Validation sets

y = pd.DataFrame(df['Ventes'])

X = df.drop(['Ventes'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, \
random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, \
test_size = 0.15, random_state=38)

features = []

print("augment features")
build_features(features, X_train)
build_features([], X_test)
print(features)

print('training data processed')

params = {"objective": "reg:linear",
           "booster" : "gbtree",
           "eta": 0.03,
           "max_depth": 6,
           "subsample": 0.8,
           "colsample_bytree": 0.7,
           "silent": 1,
           "seed": 300,
           "lambda" : 0.3
           }
num_boost_round = 1000

#params = {"objective": "reg:linear",
#          "booster" : "gblinear",
#          "lambda" : 0.01
#          }
#num_boost_round = 30000

print("Train a XGBoost model")
#y_valid = np.log1p(y_valid)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=2000, verbose_eval=True)
  
# feval=rmspe_xg,
  
print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(y_valid.Ventes.values, yhat)
#error = rmspe((y_valid), (yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(X_test[features])
test_probs = gbm.predict(dtest)
error_test = rmspe(y_test.Ventes.values, test_probs)
#error_test = rmspe((y_test), (test_probs))
print('RMSPE: {:.6f}'.format(error_test))

print("Make predictions on the test set")
dtest = xgb.DMatrix(X_test[features])
test_probs = gbm.predict(dtest)
error_test = rmspe(y_test.Ventes.values, test_probs)
#error_test = rmspe((y_test), (test_probs))
print('RMSPE: {:.6f}'.format(error_test))

F = pd.read_csv('/Anaconda3/Forecast.csv', sep=';')

print("Make actual predictions")
dforecast = xgb.DMatrix(F[features])
forecast_probs = gbm.predict(dforecast)

# Export results
result = pd.DataFrame({'Code_Produit': X_test['Code_Produit'], 'Nb_Jours': \
X_test['Nb_Jours'], 'Type_Offre': X_test['Type_Offre'], \
'Taux_Degradation' : X_test['Taux_Degradation'], 'Ventes': np.expm1(test_probs)})
true_result = pd.DataFrame({'Code_Produit': X_test['Code_Produit'], \
'Nb_Jours': X_test['Nb_Jours'], 'Type_Offre' : X_test['Type_Offre'], \
'Taux_Degradation' : X_test['Taux_Degradation'], 'Ventes': y_test['Ventes']})
forecast = pd.DataFrame({'Code_Produit': F['Code_Produit'], \
'Semaine début': F['S_debut'], 'Ventes': forecast_probs,'Magasin': F['Magasin'] })
result.to_csv("xgboost_10_Promo_submission.csv", index=False)
true_result.to_csv("xgboost_11_Promo_submission.csv", index=False)
result.to_excel("xgboost_10_Promo_submission.xlsx", index=False)
true_result.to_excel("xgboost_11_Promo_submission.xlsx", index=False)
forecast.to_excel("/Anaconda3/xgboost_12_Promo_submission.xlsx", index=False)

# XGB feature importances
create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df2 = pd.DataFrame(importance, columns=['feature', 'fscore'])
df2['fscore'] = df2['fscore'] / df2['fscore'].sum()

featp = df2.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
