# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import pandas as pd
import numpy as np

normal_feature = pd.read_csv('feature/feature_normal.csv')
data = np.array(normal_feature)

feature = pd.DataFrame()
for i in np.arange(0,data.shape[1],1):
    for j in np.arange(i + 1,data.shape[1],1):
        feature[str(i)+'*'+str(j)] = data[:,i]*data[:,j]
        feature[str(i)+'/'+str(j)] = np.divide(data[:,i],data[:,j])
        feature[str(i)+'+'+str(j)] = data[:,i]+data[:,j]
        feature[str(i)+'-'+str(j)] = data[:,i]-data[:,j]

feature.to_csv('feature/xx_feature.csv', index=False)