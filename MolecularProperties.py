#%%
import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import seaborn as sb

#%%
#データを読み込む
train = pd.read_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_processed.csv")
test = pd.read_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling//test_processed.csv")
# sample_submission = pd.read_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/sample_submission.csv")
structures = pd.read_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling//structures.csv")

# Win
# C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/
# Mac
# ⁨/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/


#%%
# 分子名+atom_index_0/1で原子の種類と2つの原子の座標を得る
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)

# 各分子の原子間距離を測る
train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

train['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

#%%
# 中間ファイルとして書き出す
train.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_processed.csv")
test.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_processed.csv")

#%%
# 計算用にサンプルデータを作る
train_sample = train.sample(n=1000)
# train_1JHN=train[train["type"] == "1JHN"]
# plt.hist(train_1JHN["scalar_coupling_constant"], bins=100)


#%%
# plt.scatter(train_sample["type"], train_sample["scalar_coupling_constant"])
train_group1 = train_sample[train_sample["type"] == "1JHC"]
# train_group2 = train_sample[train_sample["type"] == "3JHC"|train_sample["type"] == "3JHH"]
# train_group3 = train_sample[train_sample["type"] == "1JHN"]
# plt.hist(train_group3["scalar_coupling_constant"], bins=100)
train_group1

#%%
# データをマージ
acc_dic = {}

train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

test['scalar_coupling_constant'] = 9999999999

alldata["type"][alldata["type"] == "1JHC" ] = 0
alldata["type"][alldata["type"] == "2JHH" ] = 1
alldata["type"][alldata["type"] == "1JHN"] = 2
alldata["type"][alldata["type"] == "2JHN" ] = 3
alldata["type"][alldata["type"] == "2JHC" ] = 4
alldata["type"][alldata["type"] == "3JHH"] = 5
alldata["type"][alldata["type"] == "3JHC" ] = 6
alldata["type"][alldata["type"] == "3JHN"] = 7

alldata["atom_0"][alldata["atom_0"] == "H" ] = 1
alldata["atom_1"][alldata["atom_1"] == "H" ] = 1
alldata["atom_1"][alldata["atom_1"] == "C" ] = 6
alldata["atom_1"][alldata["atom_1"] == "N" ] = 7

alldata["molecule_name"] = alldata["molecule_name"].str[-6:].astype(int)

alldata["type"].astype(int)
alldata["atom_0"].astype(int)
alldata["atom_1"].astype(int)
# alldata["scc_class"].astype(int)

#%%
alldata.describe()

#%%
train_sample_feature = train_sample.drop(["id"], axis=1)
plt.figure(figsize=(15,15))
sb.heatmap(train_sample_feature.corr(), annot=True)


#%%
df = pd.DataFrame(test["type"].unique())
df

#%%
# train_sample = train.sample(n=100)
sb.relplot(x="type", y="scalar_coupling_constant", data=train_sample)
# plt.scatter(train_sample["molecule_name"], train_sample["scalar_coupling_constant"])

#%%
train_sample

#%%
# train.describe()
# test.describe()
# structures.describe()
plt.hist(train['scalar_coupling_constant'], bins=100)


#%%

# 訓練データ特徴量をリスト化
cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()
num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()

other_cols = ['id','WhatIsData']
# 余計な要素をリストから削除
cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去
num_cols.remove('id') #Id削除

# カテゴリカル変数をダミー化
cat = pd.get_dummies(alldata[cat_cols])

# データ統合
all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)

# plt.hist(np.log(train['revenue']), bins=50)
# plt.hist(train['revenue'], bins=50)

train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','id'], axis=1).reset_index(drop=True)
test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','scalar_coupling_constant'], axis=1).reset_index(drop=True)

x_ = train_.drop('scalar_coupling_constant',axis=1)
y_ = train_.loc[:, ['scalar_coupling_constant']]
test_feature = test_.drop('id',axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    x_, y_, test_size=0.33, random_state=42)

#%%
# RandomForestRegressorによる予測
forest_parameters = {'n_estimators': [1000]}

# clf = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1)
clf = GridSearchCV(RandomForestRegressor(), forest_parameters, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
best = clf.best_estimator_

# acc_forest = clf.score(X_train, y_train)
# acc_dic.update(model_forest = round(acc_forest,3))
# print(f"training dataに対しての精度: {clf.score(X_train, y_train):.2}")
prediction_rf_gs = clf.predict(test_feature)

#%%
best

#%%
# lightGBMによる予測
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
        'verbose' : 0,
        'n_jobs': 2
}

gbm = lgb.train(params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)

prediction_lgb = gbm.predict(test_feature)

#%%
# LGBMRegressorによる予測
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}

model = LGBMRegressor(**LGB_PARAMS, n_estimators=10000, n_jobs = -1)
model.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='mae',
        verbose=500, early_stopping_rounds=100)

prediction_lgb = model.predict(test_feature)


#%%
# RandomForestRegressorによる予測
forest = RandomForestRegressor().fit(X_train, y_train)
prediction_rf = forest.predict(test_feature)

acc_forest = forest.score(X_train, y_train)
acc_dic.update(model_forest = round(acc_forest,3))
print(f"training dataに対しての精度: {forest.score(X_train, y_train):.2}")

#%%
# lasso回帰による予測
lasso = Lasso().fit(X_train, y_train)
prediction_lasso = np.exp(lasso.predict(test_feature))

acc_lasso = lasso.score(X_train, y_train)
acc_dic.update(model_lasso = round(acc_lasso,3))
print(f"training dataに対しての精度: {lasso.score(X_train, y_train):.2}")

#%%
# ElasticNetによる予測
En = ElasticNet().fit(X_train, y_train)
prediction_en = np.exp(En.predict(test_feature))
print(f"training dataに対しての精度: {En.score(X_train, y_train):.2}")

acc_ElasticNet = En.score(X_train, y_train)
acc_dic.update(model_ElasticNet = round(acc_ElasticNet,3))

#%%
# ElasticNetによるパラメータチューニング
parameters = {
        'alpha'      : [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio'   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

En2 = GridSearchCV(ElasticNet(), parameters)
En2.fit(X_train, y_train)
prediction_en2 = np.exp(En2.predict(test_feature))

acc_ElasticNet_Gs = En2.score(X_train, y_train)
acc_dic.update(model_ElasticNet_Gs = round(acc_ElasticNet_Gs,3))
print(f"training dataに対しての精度: {En2.score(X_train, y_train):.2}")


#%%
# 各モデルの訓練データに対する精度をDataFrame化
Acc = pd.DataFrame([], columns=acc_dic.keys())
dict_array = []
for i in acc_dic.items():
        dict_array.append(acc_dic)
Acc = pd.concat([Acc, pd.DataFrame.from_dict(dict_array)]).T
Acc[0]

#%%
# Idを取得
Id = np.array(test["id"]).astype(int)
# 予測データとIdをデータフレームへ落とし込む
result = pd.DataFrame(prediction_lgb, Id, columns = ["scalar_coupling_constant"])
# csvとして書き出し
result.to_csv("⁨prediction_Molecule_lgb.csv", index_label = ["Id"])

#%%
City_unique = alldata["City"].unique()
df_city = pd.DataFrame(City_unique)
# csvとして書き出し
df_city.to_csv("CityList.csv", index_label = ["City_name"])

#%%
test_["molecule_name"].describe()

#%%
#%%
#データを読み込む
dipole_moments = pd.read_csv("C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/dipole_moments.csv")
magnetic_shielding_tensors = pd.read_csv("C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/magnetic_shielding_tensors.csv")
mulliken_charges = pd.read_csv("C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/mulliken_charges.csv")
potential_energy = pd.read_csv("C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/potential_energy.csv")

#%%
plt.hist(train_sample['scalar_coupling_constant'], bins=50)

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

Missing_table(test)

#%%
# サンプルからデータ型を調べる関数
def Datatype_table(df):
        list_type = df.dtypes #データ型
        Datatype_table = pd.concat([list_type], axis = 1)
        Datatype_table_len = Datatype_table.rename(columns = {0:'データ型'})
        return Datatype_table_len
    
Datatype_table(alldata)