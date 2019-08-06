#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("hp_train.csv")
test = pd.read_csv("hp_test.csv")

# train.describe()


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
# 訓練データ特徴量をリスト化
train_cat_cols = train.dtypes[train.dtypes=='object'].index.tolist()
train_num_cols = train.dtypes[train.dtypes!='object'].index.tolist()

train_cat = pd.get_dummies(train[train_cat_cols])

# データ統合
train_all_data = pd.concat([train[train_num_cols].fillna(0),train_cat],axis=1)

train_all_data.describe()

# テストデータ特徴量をリスト化
test_cat_cols = test.dtypes[train.dtypes=='object'].index.tolist()
test_num_cols = test.dtypes[train.dtypes!='object'].index.tolist()

test_cat = pd.get_dummies(test[test_cat_cols])

# データ統合
test_all_data = pd.concat([test[test_num_cols].fillna(0),test_cat],axis=1)

test_all_data.describe()

#%%
# lasso回帰による予測

x_ = train_all_data.drop('SalePrice',axis=1)
y_ = train_all_data.loc[:, ['SalePrice']]


#%%
# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
result = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
# my_tree_one.csvとして書き出し
result.to_csv("prediction_forest.csv", index_label = ["PassengerId"])

#%%
