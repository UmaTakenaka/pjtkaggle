#%%
import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score

train = pd.read_csv("rest_train.csv")
test = pd.read_csv("rest_test.csv")


#%%
# サンプルから欠損値と割合、データ型を調べる関数
def Missing_table(df):
    null_val = df.isnull().sum()
    # null_val = df.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
    percent = 100 * null_val/len(df)
    # list_type = df.isnull().sum().dtypes #データ型
    Missing_table = pd.concat([null_val, percent], axis = 1)
    missing_table_len = Missing_table.rename(
    columns = {0:'欠損値', 1:'%', 2:'type'})
    return missing_table_len.sort_values(by=['欠損値'], ascending=False)

Missing_table(train)

#%%
# サンプルからデータ型を調べる関数
def Datatype_table(df):
        list_type = df.dtypes #データ型
        Datatype_table = pd.concat([list_type], axis = 1)
        Datatype_table_len = Datatype_table.rename(columns = {0:'データ型'})
        return Datatype_table_len
    
Datatype_table(train)

#%%
train[['Open Date', 'City', 'City Group', 'Type']].describe()
# pd.pivot_table(train, index='City', columns='City Group')
# plt.hist(np.log(train['revenue']), bins=20)

#%%
train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
test['revenue'] = 9999999999
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

alldata["Open Date"] = pd.to_datetime(alldata["Open Date"])
alldata["Year"] = alldata["Open Date"].apply(lambda x:x.year)
alldata["Month"] = alldata["Open Date"].apply(lambda x:x.month)
alldata["Day"] = alldata["Open Date"].apply(lambda x:x.day)

alldata = alldata.drop('Open Date', axis=1)

#%%
# alldata.head()
Datatype_table(alldata)

#%%
# 訓練データ特徴量をリスト化
cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()
num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()

other_cols = ['Id','WhatIsData']
# 余計な要素をリストから削除
cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去
num_cols.remove('Id') #Id削除

# カテゴリカル変数をダミー化
cat = pd.get_dummies(alldata[cat_cols])

# データ統合
all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)

# plt.hist(np.log(train['revenue']), bins=50)
# plt.hist(train['revenue'], bins=50)
#%%
all_data.head()

#%%
# lightGBMによる予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)

x_ = train_.drop('revenue',axis=1)
y_ = train_.loc[:, ['revenue']]
y_ = np.log(y_)
test_feature = test_.drop('Id',axis=1)

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
            num_boost_round=100,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)

# print(f"training dataに対しての精度: {gbm.score(x_, y_):.2}")
prediction = np.exp(gbm.predict(test_feature))

#%%
# RandomForestRegressorによる予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)

x_ = train_.drop('revenue',axis=1)
y_ = train_.loc[:, ['revenue']]
y_ = np.log(y_)

forest = RandomForestRegressor().fit(x_, y_)
print(f"training dataに対しての精度: {forest.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = np.exp(forest.predict(test_feature))

#%%
# lasso回帰による予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)

x_ = train_.drop('revenue',axis=1)
y_ = train_.loc[:, ['revenue']]
# y_ = np.log(y_)

lasso = Lasso().fit(x_, y_)
print(f"training dataに対しての精度: {lasso.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = np.exp(lasso.predict(test_feature))

#%%
# ElasticNetによる予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)

x_ = train_.drop('revenue',axis=1)
y_ = train_.loc[:, ['revenue']]
y_ = np.log(y_)

En = ElasticNet().fit(x_, y_)
print(f"training dataに対しての精度: {En.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = np.exp(En.predict(test_feature))

#%%
# ElasticNetによるパラメータチューニング
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)

x_ = train_.drop('revenue',axis=1)
y_ = train_.loc[:, ['revenue']]
y_ = np.log(y_)

parameters = {
        'alpha'      : [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio'   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

En = GridSearchCV(ElasticNet(), parameters)
En.fit(x_, y_)
print(f"training dataに対しての精度: {En.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = np.exp(En.predict(test_feature))


#%%
# Idを取得
Id = np.array(test["Id"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
result = pd.DataFrame(prediction, Id, columns = ["Prediction"])
# my_tree_one.csvとして書き出し
result.to_csv("prediction_Restaurant.csv", index_label = ["Id"])

#%%
