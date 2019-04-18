import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
import time
import random
from pandas import DataFrame as DF
from sklearn.ensemble import GradientBoostingRegressor as GBR
import xgboost as xgb
from sklearn.cross_validation import cross_val_score as cv

'''
模拟退火算法计算  最佳特征的选择
'''

train = pd.read_csv('./data/do_feat_data_2/train.csv')
test = pd.read_csv('./data/do_feat_data_2/test.csv')
# def get_model(nums, cv_fold):
#     '''
#     这部分是指定迭代次数和  cv次数下的 多模型融合算法  传入参数可以得到多模型融合的结果。
#
#     使用了 LGBMClassifier   XGBClassifier   GBC 三个模型分别做cv后再做融合
#     '''
#     feature_name1 = train_data[feature_name].columns
#     get_ans_face = list(set(get_pic(gbc_model, feature_name1).head(nums)['name']) & set(
#         get_pic(xgb_model, feature_name1).head(nums)['name']) & set(
#         get_pic(lgb_model, feature_name1).head(nums)['name']))
#     print('New Feature: ', len(get_ans_face))
#     if 'SNP32*SNP34' not in get_ans_face:
#         get_ans_face.append('SNP32*SNP34')
#     print('New Feature: ', len(get_ans_face))
#     new_lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=300, max_depth=3, min_child_samples=6,
#                                        learning_rate=0.102, random_state=1)
#     cv_model = cv(new_lgb_model, train_data[get_ans_face], train_label, cv=cv_fold, scoring='f1')
#     new_lgb_model.fit(train_data[get_ans_face], train_label)
#     m1 = cv_model.mean()
#
#     new_xgb_model1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, max_depth=4, learning_rate=0.101,
#                                        random_state=1)
#     cv_model = cv(new_xgb_model1, train_data[get_ans_face].values, train_label, cv=cv_fold, scoring='f1')
#     new_xgb_model1.fit(train_data[get_ans_face].values, train_label)
#     m2 = cv_model.mean()
#
#     new_gbc_model = GBC(n_estimators=310, subsample=1, min_samples_split=2, max_depth=3, learning_rate=0.1900,
#                         min_weight_fraction_leaf=0.1)
#     kkk = train_data[get_ans_face].fillna(7)
#     cv_model = cv(new_gbc_model, kkk[get_ans_face], train_label, cv=cv_fold, scoring='f1')
#     new_gbc_model.fit(kkk.fillna(7), train_label)
#
#     m3 = cv_model.mean()
#     print((m1 + m2 + m3) / 3)
#     pro1 = new_lgb_model.predict_proba(test_data[get_ans_face])
#     pro2 = new_xgb_model1.predict_proba(test_data[get_ans_face].values)
#     pro3 = new_gbc_model.predict_proba(test_data[get_ans_face].fillna(7).values)
#     ans = (pro1 + pro2 + pro3) / 3
#     return ans


# temp = [140,160,180,200,220,240,260,280,300,320]

# ans = []
# for i in range(len(temp)):
#     print('Now All Feature:',temp[i])
#     ans = get_model(temp[i],5)
#     if i == 0:
#         ans1 = ans
#     else:
#         ans1 += ans
# ans1 /= len(temp)

def find_best_feature(feature_name, cv_fold,train_data,train_label):
    # 为了寻找最佳的特征组合，这里是对LGBMClassifier  XGBClassifier   GBC三个模型的得分进行平均，来代表这个特征所代表的分数
    get_ans_face = feature_name
    new_lgb_model = lgb.LGBMRegressor(n_estimators=300,random_state=1)
    cv_model = cv(new_lgb_model, train_data[get_ans_face], train_label, cv=cv_fold, scoring='r2')
    new_lgb_model.fit(train_data[get_ans_face], train_label)
    m1 = cv_model.mean()

    new_xgb_model1 = xgb.XGBRegressor(n_estimators=300,random_state=1)
    cv_model = cv(new_xgb_model1, train_data[get_ans_face].values, train_label, cv=cv_fold, scoring='r2')
    new_xgb_model1.fit(train_data[get_ans_face].values, train_label)
    m2 = cv_model.mean()

    new_gbc_model = GBR(n_estimators=310)
    cv_model = cv(new_gbc_model, train_data[get_ans_face].values, train_label, cv=cv_fold, scoring='r2')
    new_gbc_model.fit(train_data[get_ans_face].values, train_label)
    m3 = cv_model.mean()
    return (m1 + m2 + m3) / 3
#
#
# def train_best_feature(feature_name):
#     # 这里是三个模型的预测结果取平均
#     get_ans_face = feature_name
#     new_lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=300, max_depth=3, min_child_samples=6,
#                                        learning_rate=0.102, random_state=1)
#     new_lgb_model.fit(train_data[get_ans_face], train_label)
#
#     new_xgb_model1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, max_depth=4, learning_rate=0.101,
#                                        random_state=1)
#     new_xgb_model1.fit(train_data[get_ans_face].values, train_label)
#
#     new_gbc_model = GBC(n_estimators=310, subsample=1, min_samples_split=2, max_depth=3, learning_rate=0.1900,
#                         min_weight_fraction_leaf=0.1)
#     kkk = train_data[get_ans_face].fillna(7)
#     new_gbc_model.fit(kkk.fillna(7), train_label)
#
#     pro1 = new_lgb_model.predict_proba(test_data[get_ans_face])
#     pro2 = new_xgb_model1.predict_proba(test_data[get_ans_face].values)
#     pro3 = new_gbc_model.predict_proba(test_data[get_ans_face].fillna(7).values)
#     ans = (pro1 + pro2 + pro3) / 3
#
#     return ans


def get_pic(model, feature_name):
    ans = DF()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    #     print(ans[ans['score']>0].shape)
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

def get_start():
    '''
    :return:使用单模型做5折cv
    '''
    lgbm_submission = pd.read_csv('./data/base_data/submit_example.csv',header=None)
    lgbm_submission.columns=['A','B']
    print('###:',)
    #train_flg = pd.read_csv(os.getcwd() + '/data/train/train_flg.csv', sep='\t')
    train = pd.read_csv( './data/do_feat_data_2/train.csv')
    test = pd.read_csv( './data/do_feat_data_2/test.csv')
    train.pop('板温**1/2')
    train.pop('现场温度**1/2')
    test.pop('板温**1/2')
    test.pop('现场温度**1/2')
    train=train.interpolate()

    train=train[~train.isin([np.nan,np.inf,-np.inf]).any(1)]

    # is_nan_air = np.isnan(train).any()
    # print(is_nan_air)
    # for index,i in enumerate(list(is_nan_air)):
    #     if(i==True):
    #         print('::::',train.iloc[:,index])

    y=train.pop('发电量')





    lgb_model = lgb.LGBMRegressor(n_estimators=120)
    # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
    lgb_model.fit(train, y)

    #



    xgb_model = xgb.XGBRegressor(n_estimators=120)
    # cv_model = cv(xgb_model, train_data[feature_name].values, train_label,  cv=10, scoring='f1')
    xgb_model.fit(train, y)


    gbc_model = GBR(n_estimators=200)
    # cv_model = cv(gbc_model, kkk[feature_name], train_label,  1cv=10, scoring='f1')
    gbc_model.fit(train, y)


    nums = 100
    feature_name1 = train.columns
    #这里的|运算是分别获取lgb_model 认为最重要的45个特征，xgb_model、gbc_model等各自认为最重要的45个特征进行
    get_ans_face = list(set(get_pic(lgb_model,feature_name1).head(nums)['name'])|set(get_pic(xgb_model,feature_name1).head(nums)['name'])|set(get_pic(gbc_model,feature_name1).head(nums)['name']))
    print('New Feature: ',len(get_ans_face))


    now_feature = []
    check = 0
    for i in range(len(get_ans_face)):
        #贪心方法找到最佳的feature组合     这里是使用不断添加feature的方式，
        now_feature.append(get_ans_face[i])
        jj = find_best_feature(now_feature,6,train,y)
        print('vvvv:',jj)
        if jj>check:
            print('目前特征长度为',len(now_feature),' 目前帅气的cv值是',jj,' 成功加入第',i+1,'个','增值为',jj-check)
            check = jj
        else:
            now_feature.pop()
    #         print('目前特征长度为',len(now_feature),'第',i+1,'个拉闸了')
    #



    train[now_feature].to_csv('./data/do_feat_data/train_tuihuo.csv')
if __name__=='__main__':
    get_start()