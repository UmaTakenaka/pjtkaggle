#%%
train_below = train[train['scalar_coupling_constant'] < 50]
# plt.hist(train_below['scalar_coupling_constant'], bins=100)
train_upper = train[train['scalar_coupling_constant'] >= 50]
plt.hist(train_upper['scalar_coupling_constant'], bins=100)

#%%