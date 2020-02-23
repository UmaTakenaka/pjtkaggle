#%%

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt

#データを読み込む
print('Loading dataset...')
train = pd.read_csv("/Users/yumatakenaka/Data/ratings_sample.csv")
print('Finished')

# trainをランダムサンプリング
# train = train.sample(frac=0.01)
# カラムをカテゴリ変数化
userId_categorical = pd.api.types.CategoricalDtype(categories=sorted(train.userId.unique()), ordered=True)
movieId_categorical = pd.api.types.CategoricalDtype(categories=sorted(train.movieId.unique()), ordered=True)
# カテゴリインスタンスを利用して新しいカラムを作成
row = train.userId.astype(userId_categorical).cat.codes
col = train.movieId.astype(movieId_categorical).cat.codes
# マトリックスにRatingの数値を当てはめる
sparse_matrix = csr_matrix((train["rating"], (row, col)), shape=(userId_categorical.categories.size, movieId_categorical.categories.size))
# スパース行列をDataframe化する
train_pivot = pd.SparseDataFrame(sparse_matrix, index = userId_categorical.categories, columns = movieId_categorical.categories, default_fill_value = 0, dtype = 'int')

# %%
# n_neiborsやalgorithm、metricなど重要なアーギュメントを設定
knn = NearestNeighbors(n_neighbors=9,algorithm= 'brute', metric= 'cosine')
# 前処理したデータセットでモデルを訓練
model_knn = knn.fit(train_pivot)

# %%
def movie_prediction(movie):
    distance, indice = model_knn.kneighbors(train_pivot.iloc[train_pivot.index== movie].values.reshape(1,-1),n_neighbors=11)
    for i in range(0, len(distance.flatten())):
        if  i == 0:
            print('Recommendations if you like the movie {0}:\n'.format(train_pivot[train_pivot.index== movie].index[0]))
        else:
            print('{0}: {1} with distance: {2}'.format(i,train_pivot.index[indice.flatten() [i]],distance.flatten()[i]))

# %%
train_pivot.head()

# %%
movie_prediction(187)

# %%
