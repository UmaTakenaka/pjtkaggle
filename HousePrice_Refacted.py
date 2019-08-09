#%%
import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

#%%
train = pd.read_csv("hp_train.csv")
test = pd.read_csv("hp_test.csv")

acc_dic = {}

train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
test['SalePrice'] = 9999999999
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)


# 訓練データ特徴量をリスト化
cat_cols = alldata.dtypes[train.dtypes=='object'].index.tolist()
num_cols = alldata.dtypes[train.dtypes!='object'].index.tolist()

other_cols = ['Id','WhatIsData']
# 余計な要素をリストから削除
cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去
num_cols.remove('Id') #Id削除

cat = pd.get_dummies(alldata[cat_cols])

# データ統合
all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

# 特徴量生成
x_ = train_.drop('SalePrice',axis=1)
y_ = train_.loc[:, ['SalePrice']]
y_ = np.log(y_)
test_feature = test_.drop('Id',axis=1)


#%%
# lightGBMによる予測
X_train, X_test, y_train, y_test = train_test_split(
    x_, y_, test_size=0.33, random_state=201612
)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# LightGBM parameters
params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8,
        'bagging_freq': 5,
        'verbose' : 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)


prediction_train = gbm.predict(X_train)
y_pred = prediction_train.tolist()
y_true = y_train['SalePrice'].tolist()
acc_lgb = f1_score(y_true, y_pred)
# acc_dic.update(model_lgb = acc_lgb)
# acc_dic

prediction_lgb = np.exp(gbm.predict(test_feature))


#%%
# lasso回帰による予測
lasso = Lasso().fit(X_train, y_train)

acc_lasso = lasso.score(X_train, y_train)
acc_dic.update(model_lasso = round(acc_lasso,3))
print(f"training dataに対しての精度: {lasso.score(X_train, y_train):.2}")

prediction_lasso = lasso.predict(test_feature)


#%%
# ElasticNetによる予測
En = ElasticNet().fit(X_train, y_train)

acc_ElasticNet = En.score(X_train, y_train)
acc_dic.update(model_ElasticNet = round(acc_ElasticNet,3))
print(f"training dataに対しての精度: {En.score(X_train, y_train):.2}")

prediction_ElasticNet = np.exp(En.predict(test_feature))

#%%
# ElasticNetによるパラメータチューニング
parameters = {
        'alpha'      : [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio'   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

En = GridSearchCV(ElasticNet(), parameters)
En.fit(x_, y_)
print(f"training dataに対しての精度: {En.score(x_, y_):.2}")

prediction = np.exp(En.predict(test_feature))

#%%
Acc = pd.DataFrame([], columns=acc_dic.keys())
dict_array = []
for i in acc_dic.items():
        dict_array.append(acc_dic)
Acc = pd.concat([Acc, pd.DataFrame.from_dict(dict_array)]).T
Acc_table = Acc.drop(1, axis=1).rename(columns = {0:'Accuracy'})
Acc_table

#%%
# Idを取得
Id = np.array(test["Id"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
result = pd.DataFrame(prediction, Id, columns = ["SalePrice"])
# my_tree_one.csvとして書き出し
result.to_csv("prediction_regression.csv", index_label = ["Id"])

#%%
