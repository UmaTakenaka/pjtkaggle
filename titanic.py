#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.describe()


#%%
def Missing_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    Missing_table = pd.concat([null_val, percent], axis = 1)
    missing_table_len = Missing_table.rename(
    columns = {0:'欠損値', 1:'%'})
    return missing_table_len

Missing_table(train)

#%%
# 欠損値の埋め方
# train["Age"] = train["Age"].fillna(train["Age"].median())
train["Age"] = train["Age"].fillna(train["Age"].mean())
# train = train.dropna(subset = ["Age"])
train["Embarked"] = train["Embarked"].fillna("S")

# train["FSize"] = train["SibSp"] + train["Parch"] + 1

# Missing_table(train)
plt.hist(train["Age"], bins=20)

#%%
# 名義尺度の置き換え
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# test["Age"] = test["Age"].fillna(test["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].mean())
# test = test.dropna(subset = ["Age"])
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.mean()
# test["FSize"] = test["SibSp"] + test["Parch"] + 1


#%%
# SVMによる予測

x_ = train["Survived"].values
y_ = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

model = SVC(kernel='linear', random_state=None)
model.fit(y_, x_)
 
# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction = model.predict(test_features)

print(my_prediction)

#%%
# Random Forestによる予測

x2_ = train["Survived"].values
y2_ = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# clf = RandomForestClassifier(random_state=0)


parameters = {
        'n_estimators'      : [10,25,50,75,100],
        'random_state'      : [0],
        'n_jobs'            : [4],
        'min_samples_split' : [5,10, 15, 20,25, 30],
        'max_depth'         : [5, 10, 15,20,25,30]
}
clf = GridSearchCV(RandomForestClassifier(), parameters)

clf.fit(y2_, x2_)

# 「test」の説明変数の値を取得
feature = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction = clf.predict(feature)
print(prediction)

#%%
# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
result = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
# my_tree_one.csvとして書き出し
result.to_csv("prediction_forest.csv", index_label = ["PassengerId"])

#%%
