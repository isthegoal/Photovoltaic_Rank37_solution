import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor as XGBR
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
import time
import random
import operator
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame as DF
start_time =time.time()
time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

random.seed(1000)

'''
单模型+排名前80特征选择
'''



def my_score(estimator, X, y):
    predicted = estimator.predict(X)
    return 1/(1+np.sqrt(np.mean((predicted - y) ** 2)))

def get_pic(model, feature_name):
    ans = DF()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    #     print(ans[ans['score']>0].shape)
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

def gey_100_impot_feature(train,y,nums=300):
    print('nums:',nums)
    #
    lgb_model = XGBR(n_estimators=120)
    # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
    lgb_model.fit(train, y)
    feature_name1 = train.columns
    #print('用到的特征：：：',feature_name.ix[:nums])
    return  list(set(get_pic(lgb_model,feature_name1).head(nums)['name']))

def UseLightGBM():
    '''
    :return:使用单模型做5折cv
    '''
    lgbm_submission = pd.read_csv('./data/base_data/submit_example.csv',header=None)
    lgbm_submission.columns=['A','B']
    print('###:',)
    #train_flg = pd.read_csv(os.getcwd() + '/data/train/train_flg.csv', sep='\t')
    train = pd.read_csv( './data/do_feat_data_2/train.csv')
    test = pd.read_csv( './data/do_feat_data_2/test.csv')
    #tuihuo_train = pd.read_csv( './data/do_feat_data/train_tuihuo.csv')
    y = train.pop('发电量')

    imp_feature_nam=gey_100_impot_feature(train,y)
    print('imp_feature_nam',imp_feature_nam)


    #print(':::::::',tuihuo_train.columns)
    test=test[imp_feature_nam]
    #id是有用的
    #train.pop('ID')
    #test.pop('ID')
    print(train.columns)
    # train=np.log(train)
    # test=np.log(test)
    # print('')

    #print('train:',train['发电量'])

    train=train[imp_feature_nam]

    print('train:',train.head())
    print('test:',test.head())
    ####################################    交叉检验   #####################################
    N = 10
    xx_cv = []
    xx_pre = []
    xx_beat = {}
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    modelScores = []

    for train_index, test_index in kf.split(train):
        X_train, X_test, y_train, y_test = train.values[train_index], train.values[test_index], y[train_index], y[test_index]

        # specify your configurations as a dict

        reg = XGBR(n_estimators=120)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], verbose=True,
                early_stopping_rounds=20)



        print('Start training...')


        #y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        #将使用这个最好的得分作为xx_cv下的附加得分
        xx_cv.append(my_score(reg,X_test,y_test))
        print('得分是：',xx_cv)

        #xx_pre.append(reg.predict(test))
        #图像展示lgb模型训练时候的特征重要度,并保存起来(两种保存方式:原序和降序)
        # plot_importance(gbm)
        # plt.show()
        FeatureScore=reg.feature_importances_
        df=pd.DataFrame({'Feature':train.columns.tolist(),'importance':FeatureScore.tolist()})
        df.to_csv(os.getcwd() +'/record/xgb_feautre_important_{0}.csv'.format(str(time_date)), index=False)
        sort_df=df.sort_values(by='importance',ascending=False)
        sort_df.to_csv(os.getcwd() +'/record/sort_xgb_feautre_important_{0}.csv'.format(str(time_date)), index=False)
        # df.plot(kind='bar',title='feature important')
        # plt.show()


    #获取最好cv得分的序号，使用这个序号去找到序号下对应的预测结果
    sorted_cv = sorted(enumerate(xx_cv), key=lambda x: x[1], reverse=True)
    best_cv_index=sorted_cv[0][0]
    print('平均得分是：', np.mean(xx_cv))
    ####################################    直接生成概率结果   #####################################

    reg = XGBR(n_estimators=120)
    reg.fit(train, y)



    print('Start training...')
    # train

    pred_y=reg.predict(test)


    res = pd.DataFrame()
    res['A'] = list(lgbm_submission['A'])
    print('!!!!!!',len(res['A']))
    print('!222!!!!!:', len(xx_pre[best_cv_index]))
    res['B'] = list(pred_y)
    #print(res[res['RST'] == 1].shape)

    ####################################    提交和线下结果展示   #####################################
    res.to_csv(os.getcwd() +'/submit/xgb_%s_%s.csv'% (str(time_date), str(np.mean(xx_cv)).split('.')[1]),
               index=False, header=None)
    print('end tiem', time.time() - start_time)
    print('info')
    print('线下成绩约', np.mean(xx_cv))


if __name__=='__main__':
    UseLightGBM()

