# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 01:37:06 2017

@author: Admin
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
 
os.chdir("C:\\Users\\Admin\\Desktop\\mlware2")
dir=os.getcwd()
pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv(os.path.join(dir,'train.csv'))
test = pd.read_csv(os.path.join(dir,'test.csv'))
matrix = pd.concat([train,test]).pivot('userId','itemId','rating')
movie_means = matrix.median()
user_means = matrix.median(axis=1)
mzm = matrix-movie_means
mz = mzm.fillna(0)
mask = -mzm.isnull()




iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=2,random_state=4)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+movie_means
m = m.clip(lower=1,upper=10)

test['rating'] = test.apply(lambda x:m[m.index==x.userId][x.itemId].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating.isnull())[0]
test.ix[missing,'rating'] = user_means[test.loc[missing].userId].values

test.to_csv('submission_SV.csv',index=False,columns=['ID','rating'])