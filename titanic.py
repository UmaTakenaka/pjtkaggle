#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

#%%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    kesson_table = pd.concat([null_val, percent], axis = 1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0:'欠損値', 1:'%'})
    return kesson_table_ren_columns

kesson_table(train)

#%%
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

kesson_table(train)


#%%

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
 
train.head(10)

#%%

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()
 
test.head(10)

#%%
# SVMによる予測

target = train["Survived"].values
features_one = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

model = SVC(kernel='linear', random_state=None)
model.fit(features_one, target)
 
# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction = model.predict(test_features)

print(my_prediction)

#%%
# Random Forestによる予測

target2 = train["Survived"].values
features_one2 = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

clf = RandomForestClassifier(random_state=0)
clf = clf.fit(features_one2, target2)
# 「test」の説明変数の値を取得
test_features2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction2 = clf.predict(test_features2)
print(my_prediction2)

#%%
# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction2, PassengerId, columns = ["Survived"])
# my_tree_one.csvとして書き出し
my_solution.to_csv("my_forest_one.csv", index_label = ["PassengerId"])

#%

