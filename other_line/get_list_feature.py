# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import pandas as pd
import numpy as np

train = pd.read_csv('data/public.train.csv')
test = pd.read_csv('data/public.test.csv')
data = pd.concat([train, test])

del data['ID'], data['发电量']

data.to_csv('feature/feature_normal.csv', index=False)
