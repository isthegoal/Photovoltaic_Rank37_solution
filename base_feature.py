import pandas as pd
import numpy as np
from pandas import DataFrame as DF
def get_division_feature(data ,feature_name):
    # 创造出特征之间 进行四则变换的特征， 每两两特征之间进行变化，  并且进行变换之后把变化后的 新特证名记录下来，（知道了吧。不想之前直接一起，
    # 特征名都完全丢失了
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns ) -1):
        for j in range( i +1 ,len(data[feature_name].columns)):
            # 保存新创建的特征值和特征名
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '+' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '-' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i] ] /data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] *data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] +data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] -data[data[feature_name].columns[j]])


    temp_data = DF(pd.concat(new_feature ,axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data ,temp_data] ,axis=1).reset_index(drop=True)

    print(data.shape)

    return data.reset_index(drop=True)


def get_square_feature(data, feature_name):
    # 对特征进行二项变换，开放等方式， 并将处理后的特征名记录
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns)):
        new_feature_name.append(data[feature_name].columns[i] + '**2')
        new_feature_name.append(data[feature_name].columns[i] + '**1/2')
        new_feature.append(data[data[feature_name].columns[i]] ** 2)
        new_feature.append(data[data[feature_name].columns[i]] ** (1 / 2))

    temp_data = DF(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data, temp_data], axis=1).reset_index(drop=True)

    print(data.shape)

    return data.reset_index(drop=True)

if __name__=='__main__':
    train = pd.read_csv( './data/do_data_pred/train.csv')
    test = pd.read_csv( './data/do_data_pred/test.csv')
    y_target = train.pop('发电量')

    ###################   1. 转换率、电压、电流、功率再造特征   ###################
    #平均数值
    # train['平均转换效率']=(train['转换效率A']+train['转换效率B']+train['转换效率C'])/3
    # train['平均电压']=(train['电压A']+train['电压B']+train['电压C'])/3
    # train['平均电流']=(train['电流A']+train['电流B']+train['电流C'])/3
    # train['平均功率']=(train['功率A']+train['功率B']+train['功率C'])/3
    #
    #
    # test['平均转换效率']=(test['转换效率A']+test['转换效率B']+test['转换效率C'])/3
    # test['平均电压']=(test['电压A']+test['电压B']+test['电压C'])/3
    # test['平均电流']=(test['电流A']+test['电流B']+test['电流C'])/3
    # test['平均功率']=(test['功率A']+test['功率B']+test['功率C'])/3
    ###################   2. 与平均值的差距特征   ###################
    # for i in train.columns:
    #     train['与均值差'+i]=train[i]-np.mean(train[i])
    # for i in test.columns:
    #     test['与均值差'+i]=test[i]-np.mean(test[i])
    ###################   3. 周期特征（距离波底的距离）   ###################
    #自造周期特征 计算规则，依据于 光照强度





    ###################   4. 变幻和交叉特征   ###################

    feature_name = [i for i in train.columns]
    #feature_SNP = [i for i in feature_name]

    #feature_name = [i for i in train_data.columns]
    train_data = get_division_feature(train, feature_name)
    test_data = get_division_feature(test, feature_name)
    temp_data = DF(y_target)
    temp_data.columns = ['发电量']
    print(train_data.columns)
    print('特征间变换')
    train_data = get_square_feature(train_data, feature_name)
    test_data = get_square_feature(test_data, feature_name)
    train = pd.concat([train_data, temp_data], axis=1).reset_index(drop=True)
    train.to_csv('./data/do_feat_data_3/train.csv')
    test_data.to_csv('./data/do_feat_data_3/test.csv')


