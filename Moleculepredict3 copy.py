#%%
import pandas as pd
import numpy as np

import math
import gc
import copy

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
# import seaborn as sns

# from lightgbm import LGBMRegressor

# Win
# C:/Users/takenaka.yuma/KaggleFiles/champs-scalar-coupling/
# Mac
# ⁨/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/
# Azure RDP
# C:/KaggleFiles/champs-scalar-coupling/

#%%
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}

#%%
train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
train_csv = pd.read_csv(f'C:/KaggleFiles/champs-scalar-coupling//train.csv', index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]

test_csv = pd.read_csv(f'C:/KaggleFiles/champs-scalar-coupling/test.csv', index_col='id', dtype=train_dtypes)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')
test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
structures_csv = pd.read_csv("C:/KaggleFiles/champs-scalar-coupling/structures.csv", dtype=structures_dtypes)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')

submission_csv = pd.read_csv("C:/KaggleFiles/champs-scalar-coupling//sample_submission.csv", index_col='id')

#%%
def get_index(some_csv, coupling_type):
    index_array = some_csv[some_csv["type"] == coupling_type].index.values
    df_index = pd.DataFrame(index_array).rename(columns={0:'id'})
    return df_index

def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
    return base, structures

def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df

def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df

def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
    return df

def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

def add_distance_to_center(df):
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) + 
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))

def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))
def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)
def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)
def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):
    base, structures = build_type_dataframes(some_csv, structures_csv, coupling_type)
    base = add_coordinates(base, structures, 0)
    base = add_coordinates(base, structures, 1)
    
    base = base.drop(['atom_0', 'atom_1'], axis=1)
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)
        
    add_center(atoms)
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    atoms = merge_all_atoms(atoms, structures)
    
    add_distance_to_center(atoms)
    
    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()
    
    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')
            
    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')
    
    full = add_atoms(base, atoms)
    add_distances(full)
    
    full.sort_values('id', inplace=True)
    
    return full

def take_n_atoms(df, n_atoms, four_start=4):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    return df[labels]

def build_x_y_data(some_csv, coupling_type, n_atoms):
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)
    
    df = take_n_atoms(full, n_atoms)
    df = df.fillna(0)
    print(df.columns)

    return df

#%%
train_df_group1 = build_x_y_data(train_csv, "1JHN", 7)
train_df_group2 = build_x_y_data(train_csv, "1JHC", 10)
train_df3 = build_x_y_data(train_csv, "2JHC", 9)
train_df4 = build_x_y_data(train_csv, "2JHH", 9)
train_df5 = build_x_y_data(train_csv, "2JHN", 9)
train_df6 = build_x_y_data(train_csv, "3JHC", 9)
train_df7 = build_x_y_data(train_csv, "3JHH", 10)
train_df8 = build_x_y_data(train_csv, "3JHN", 10)
train_df_group3 = pd.concat([train_df3,train_df4,train_df5,train_df6,train_df7,train_df8])

test_df_group1 = build_x_y_data(test_csv, "1JHN", 7)
index_df_group1 = get_index(test_csv, "1JHN")
test_df_group2 = build_x_y_data(test_csv, "1JHC", 10)
index_df_group2 = get_index(test_csv, "1JHC")

test_df3 = build_x_y_data(test_csv, "2JHC", 9)
index_df3 = get_index(test_csv, "2JHC")
test_df4 = build_x_y_data(test_csv, "2JHH", 9)
index_df4 = get_index(test_csv, "2JHH")
test_df5 = build_x_y_data(test_csv, "2JHN", 9)
index_df5 = get_index(test_csv, "2JHN")
test_df6 = build_x_y_data(test_csv, "3JHC", 9)
index_df6 = get_index(test_csv, "3JHC")
test_df7 = build_x_y_data(test_csv, "3JHH", 10)
index_df7 = get_index(test_csv, "3JHH")
test_df8 = build_x_y_data(test_csv, "3JHN", 10)
index_df8 = get_index(test_csv, "3JHN")

test_df_group3 = pd.concat([test_df3,test_df4,test_df5,test_df6,test_df7,test_df8])
index_df_group3 = pd.concat([index_df3,index_df4,index_df5,index_df6,index_df7,index_df8])

#%%
train_df_group1.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group1.csv")
train_df_group2.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group2.csv")
train_df_group3.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group3.csv")

test_df_group1.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group1.csv")
test_df_group2.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group2.csv")
test_df_group3.to_csv("/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group3.csv")

#%%
# train_df_group1 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group1.csv')
# train_df_group2 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group2.csv')
# train_df_group3 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group3.csv')
train_df_group3.fillna(0, inplace=True)

# test_df_group1 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group1.csv')
# test_df_group2 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group2.csv')
# test_df_group3 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/test_group3.csv')
test_df_group3.fillna(0, inplace=True)

#%%
# train_df_group1.drop('Unnamed: 0', axis=1)
train_df_group2.drop('Unnamed: 0', axis=1)
# train_df_group3.drop('Unnamed: 0', axis=1)

# test_df_group1.drop('Unnamed: 0', axis=1)
test_df_group2.drop('Unnamed: 0', axis=1)
# test_df_group3.drop('Unnamed: 0', axis=1)

#%%
# X_data_group1 = train_df_group1.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
# y_data_group1 = train_df_group1['scalar_coupling_constant'].values.astype('float32')
# test_feature_group1 = test_df_group1

X_data_group2 = train_df_group2.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
y_data_group2 = train_df_group2['scalar_coupling_constant'].values.astype('float32')
test_feature_group2 = test_df_group2

# X_data_group3 = train_df_group3.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
# y_data_group3 = train_df_group3['scalar_coupling_constant'].values.astype('float32')
# test_feature_group3 = test_df_group3

#%%
params = {'n_estimators'  : [100], 'n_jobs': [-1]}
forest = RandomForestRegressor()
model = GridSearchCV(forest, params, cv = 5)

# X_train_group1, X_test_group1, y_train_group1, y_test_group1 = train_test_split(
#     X_data_group1 , y_data_group1 , test_size=0.2, random_state=128)
X_train_group2, X_test_group2, y_train_group2, y_test_group2 = train_test_split(
    X_data_group2 , y_data_group2 , test_size=0.2, random_state=128)
# X_train_group3, X_test_group3, y_train_group3, y_test_group3 = train_test_split(
#     X_data_group3 , y_data_group3 , test_size=0.2, random_state=128)


# model.fit(X_train_group1, y_train_group1)
# prediction_group1 = model.predict(test_feature_group1)

print('Start Fitting')
model.fit(X_train_group2, y_train_group2)
print('Start Predicting')
prediction_group2 = model.predict(test_feature_group2)

# print('Start Fitting')
# model.fit(X_train_group3, y_train_group3)
# print('Start Predicting')
# prediction_group3 = model.predict(test_feature_group3)

# index_df_group1['scalar_coupling_constant'] = prediction_group1
# index_df_group1.to_csv("index_df_group1.csv")
index_df_group2['scalar_coupling_constant'] = prediction_group2
index_df_group2.to_csv("index_df_group2.csv")
# index_df_group3['scalar_coupling_constant'] = prediction_group3
# index_df_group3.to_csv("index_df_group3.csv")

#%%
plt.hist(index_df_group2)


#%%
# np.savetxt('out_group1.csv',prediction_group1,delimiter=',')
# np.savetxt('out_group2.csv',prediction_group2,delimiter=',')
# np.savetxt('out_group3.csv',prediction_group3,delimiter=',')

#%%
# def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms):
#     print(f'*** Training Model for {coupling_type} ***')
    
#     X_, y_, groups = build_x_y_data(train_csv, coupling_type, n_atoms)
#     test_feature, _, __ = build_x_y_data(test_csv, coupling_type, n_atoms)
#     y_pred = np.zeros(test_feature.shape[0], dtype='float32')

#     X_train, X_test, y_train, y_test = train_test_split(
#     X_, y_, test_size=0.2, random_state=128)

    # params = {'n_estimators'  : [1000], 'n_jobs': [-1]}

    # forest = RandomForestRegressor()
    # model = GridSearchCV(forest, params, cv = 5)
    # model.fit(X_train, y_train)
    # y_pred += model.predict(test_feature)

    # submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred
    
    # return y_pred

#%%
df_group1 = build_x_y_data(train_csv, "1JHN", 7)
df_group2 = build_x_y_data(train_csv, "1JHC", 10)
df3 = build_x_y_data(train_csv, "2JHC", 9)
df4 = build_x_y_data(train_csv, "2JHH", 9)
df5 = build_x_y_data(train_csv, "2JHN", 9)
df6 = build_x_y_data(train_csv, "3JHC", 9)
df7 = build_x_y_data(train_csv, "3JHH", 10)
df8 = build_x_y_data(train_csv, "3JHN", 10)
df_group3 = pd.concat([df3,df4,df5,df6,df7,df8])

#%%
# model_params = {
#     '1JHN': 7,
#     '1JHC': 10,
#     '2JHH': 9,
#     '2JHN': 9,
#     '2JHC': 9,
#     '3JHH': 9,
#     '3JHC': 10,
#     '3JHN': 10
# }

# submission = submission_csv.copy()
# for coupling_type in model_params.keys():
#     train_and_predict_for_one_coupling_type(
#         coupling_type, submission, n_atoms=model_params[coupling_type])
# submission.to_csv("⁨prediction_Molecule_rf2.csv", index_label = ["id"])




#%%
train_df_group3 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group3.csv')

#%%
train_df_group1 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group1.csv')
train_df_group2 = pd.read_csv(f'/Users/yumatakenaka/KaggleFiles/champs-scalar-coupling/train_group2.csv')

#%%
train_df_group3
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

Missing_table(train_df_group3)

#%%
train_df_group3.fillna(0,inplace=True)
#%%
index_df_group1


#%%
index_df_group1 = get_index(test_csv, "1JHN")

#%%
test_df_group3.head()

#%%
