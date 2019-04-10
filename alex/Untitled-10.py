
#%%
import numpy as np
import pandas as pd
import lightgbm as lgb


#%%
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization


#%%
import matplotlib.pyplot as plt
import seaborn as sns


#%%
import os

#%% [markdown]
# ### load data

#%%
main_folder = r"D:\dev\SanTransakt"
data_folder = os.path.join(main_folder, 'data')


#%%
#train = pd.read_csv(os.path.join(data_folder,"train_simpified.csv"))
#test = pd.read_csv(os.path.join(data_folder,"test_simpified.csv"))
train = pd.read_csv(os.path.join(data_folder,"train.csv"))
test = pd.read_csv(os.path.join(data_folder,"test.csv"))

#%% [markdown]
# ### prep data

#%%
test['target'] = np.nan
cols = test.columns[:-1].tolist()
cols = [cols[0], 'target']+cols[1:]
test = test.loc[:,cols]


#%%
traintest = pd.concat([train, test])
traintest.reset_index(drop=True, inplace=True)
traintest['set'] = 'test'
traintest.loc[train.index,'set'] = 'train'
traintest.set.value_counts()


#%%
X_tr = train.drop(columns=['target', 'ID_code']).copy()
X_te = test.drop(columns=['target', 'ID_code']).copy()
X = traintest.drop(columns=['target', 'ID_code']).copy()


#%%
y_tr = train.target.copy()
y_te = test.target.copy()
y = traintest.target.copy()

#%% [markdown]
# ### select part of the training data for a less greedy exploration

#%%
from sklearn.model_selection import train_test_split


#%%
#X_tr, X_tr_2, y_tr, y_tr_2 = train_test_split(X_tr, y_tr, test_size=0.9, random_state=42)
#X_te, X_tr_2, y_te, y_tr_2 = train_test_split(X_tr_2, y_tr_2, test_size=0.9, random_state=42)


#%%
#X_tr, X_te, y_tr, y_te = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)

#%% [markdown]
# ### train GBM model

#%%
n_folds = 12
random_seed = 6
folds = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=10)


#%%
oof = np.zeros(len(X_tr))
y_te_pred = np.zeros(len(X_te))

#%% [markdown]
# #### create model

#%%
param = {
    'bagging_freq': 5,          
    'bagging_fraction': 0.38,   'boost_from_average':'false',   
    'boost': 'gbdt',             'feature_fraction': 0.04,     'learning_rate': 0.0085,
    'max_depth': -1,             'metric':'auc',                'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,            'num_threads': 8,              'tree_learner': 'serial',   'objective': 'binary',
    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3603427518866501,'verbosity': 1
}

#%%

def label_encoder(input_df, encoder_dict):
    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    return input_df, categorical_feats.tolist(), encoder_dict

train, categorical_feats, encoder_dict = label_encoder(train, None)

X1 = train.drop('target', axis=1)
y1 = train.target

#%%
train_data = lgb.Dataset(data=X1, label=y1, categorical_feature=categorical_feats, free_raw_data=False)

# Bayesian optimization: parameters to be tuned
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, reg_alpha, reg_lambda):
    param['num_leaves'] = int(round(num_leaves))
    param['feature_fraction'] = max(min(feature_fraction, 1), 0)
    param['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    #param['max_depth'] = round(max_depth)
    param['reg_alpha'] = max(reg_alpha, 0)
    param['reg_lambda'] = max(reg_lambda, 0)
    #param['min_split_gain'] = min_split_gain
    #param['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(param, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
    return max(cv_result['auc-mean'])

# Set the range for each paramter
lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (8, 18),
                                        'feature_fraction': (0.01, 0.1),
                                        'bagging_fraction': (0.1, 1.0),
                                        #'max_depth': (5, 8.99),
                                        'reg_alpha': (0.0, 0.3),
                                        'reg_lambda': (0.0, 0.5)},
                                        #'min_split_gain': (0.001, 0.1),
                                        #'min_child_weight': (5, 50)},
                                         random_state=0)

#%%
# Bayesian optimization

import warnings
import time
warnings.filterwarnings("ignore")

init_round = 15
opt_round = 25
lgbBO.maximize(init_points=init_round, n_iter=opt_round)
opt_params = lgbBO.res[1]['params']

#copy content of opt_params to param
param = {**param, **opt_params}
param['num_leaves'] = int(round(param['num_leaves']))

#%% [markdown]
# #### perform cross-validation

#%%
get_ipython().run_cell_magic(u'time', u'', u'for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_tr.values, y_tr.values)):\n    \n    print("Fold {}".format(fold_))\n    \n    # set train and validation data\n    trn_data = lgb.Dataset(\n        X_tr.iloc[trn_idx]\n        ,label=y_tr.iloc[trn_idx]\n    )\n    val_data = lgb.Dataset(\n        X_tr.iloc[val_idx]\n        ,label=y_tr.iloc[val_idx]\n    )\n    \n    # train model\n    clf = lgb.train(\n        param\n        ,trn_data\n        ,1000000\n        ,valid_sets = [trn_data, val_data]\n        ,verbose_eval=5000\n        ,early_stopping_rounds = 2000\n    )\n    \n    # record predictions\n    oof[val_idx] = clf.predict(X_tr.iloc[val_idx], num_iteration=clf.best_iteration)\n    y_te_pred += clf.predict(X_te, num_iteration=clf.best_iteration) / folds.n_splits')

#%% [markdown]
# #### estimate test score from cross-validated score

#%%
print("CV score: {:<8.5f}".format(roc_auc_score(y_tr, oof)))
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = y_te_pred

#%% [markdown]
# #### save predictions

#%%
from datetime import datetime
date_today = datetime.now().strftime("%Y%m%d")


#%%
sub.to_csv(os.path.join(data_folder,"submission_{}.csv".format(name, date_today)), index=False)

#%% [markdown]
# #### save trained model

#%%
json_model = clf.dump_model()


#%%
import json
with open(os.path.join(data_folder,'model_gbm_{}.json'.format(name, date_today)), 'w+') as f:
    json.dump(json_model, f, indent=4)


#%%
# https://www.kaggle.com/jesucristo/30-lines-starter-solution-fast
# https://lightgbm.readthedocs.io/en/latest/Python-Intro.html


