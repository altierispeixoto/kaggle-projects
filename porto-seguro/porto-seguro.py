
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
import gc



params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 4
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['silent'] = True


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


print('loading files...')
train = pd.read_csv('../../data/train.csv', na_values=-1)
test = pd.read_csv('../../data/test.csv', na_values=-1)
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)

for c in train.select_dtypes(include=['float64']).columns:
    train[c] = train[c].astype(np.float32)
    test[c] = test[c].astype(np.float32)
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c] = train[c].astype(np.int8)
    test[c] = test[c].astype(np.int8)

print(train.shape, test.shape)

# xgb
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

X = train.drop(['id', 'target'], axis=1)
features = X.columns
X = X.values
y = train['target'].values
sub = test['id'].to_frame()
sub['target'] = 0

nrounds = 200  # need to change to 2000
kfold = 2  # need to change to 5
skf = StratifiedKFold(n_splits=kfold, random_state=0)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' xgb kfold: {}  of  {} : '.format(i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                          feval=gini_xgb, maximize=True, verbose_eval=100)
    sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values),
                                       ntree_limit=xgb_model.best_ntree_limit + 50) / (2 * kfold)

sub.to_csv('submission.csv', index=False, float_format='%.5f')
gc.collect()
sub.head(2)
