#%%
import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet

train = pd.read_csv("hp_train.csv")
test = pd.read_csv("hp_test.csv")

train.describe()


#%%
# サンプルから欠損値と割合、データ型を調べる関数
def Missing_table(df):
    # null_val = df.isnull().sum()
    null_val = df.isnull().sum()[train.isnull().sum()>0]
    percent = 100 * null_val/len(df)
    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化
    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型
    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)
    missing_table_len = Missing_table.rename(
    columns = {0:'欠損値', 1:'%', 2:'type'})
    return missing_table_len.sort_values(by=['欠損値'], ascending=False)

Missing_table(train)
# plt.hist(np.log(train['SalePrice']), bins=50)

#%%
train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
test['SalePrice'] = 9999999999
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

alldata.head()

#%%
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

plt.hist(np.log(train['SalePrice']), bins=50)
# plt.hist(train['SalePrice'], bins=50)


#%%
# lasso回帰による予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

x_ = train_.drop('SalePrice',axis=1)
y_ = train_.loc[:, ['SalePrice']]

lasso = Lasso().fit(x_, y_)
print(f"training dataに対しての精度: {lasso.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = lasso.predict(test_feature)

#%%
# ElasticNetによる予測
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

x_ = train_.drop('SalePrice',axis=1)
y_ = train_.loc[:, ['SalePrice']]
y_ = np.log(y_)

En = ElasticNet().fit(x_, y_)
print(f"training dataに対しての精度: {En.score(x_, y_):.2}")

test_feature = test_.drop('Id',axis=1)
prediction = np.exp(En.predict(test_feature))

#%%
# ElasticNetによるパラメータチューニング
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

x_ = train_.drop('SalePrice',axis=1)
y_ = train_.loc[:, ['SalePrice']]
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
result = pd.DataFrame(prediction, Id, columns = ["SalePrice"])
# my_tree_one.csvとして書き出し
result.to_csv("prediction_regression.csv", index_label = ["Id"])

#%%
