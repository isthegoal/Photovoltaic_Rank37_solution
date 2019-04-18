# -*- coding:utf-8 -*-

import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics


print('开始处理特征......')
# 读入score和test的id
label = pd.read_csv('data/public.train.csv', usecols=['发电量'])
path = 'data/public.test.csv'
test_all = pd.read_csv(path, encoding='utf8')


# 读入数据
df_normal = pd.read_csv('feature/feature_normal.csv')
df_id = pd.read_csv('feature/feature_id.csv')
df_xx = pd.read_csv('feature/xx_feature.csv')
# 合并特征集
df_feature = pd.concat([df_normal, df_id, df_xx], axis=1)

label = list(label['发电量'])
train_feature = df_feature[:len(label)]
test_feature = df_feature[len(label):]

train_feature = np.array(train_feature)
test_feature = np.array(test_feature)
# 切分训练
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,random_state=1017)
# train_feature = X_train
# label = Y_train

print('特征处理完毕......')


###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)


print('开始训练......')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'}
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval
                )
gbm.save_model('model/lgb_model.txt')

temp = gbm.predict(X_test)
print('特征重要性：'+ str(list(gbm.feature_importance())))


########################## 保存结果 ############################
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['ID'] = list(test_all['ID'])
df_result['Score'] = pre
df_result.to_csv('result/lgb_result.csv', index=False, header=False)

