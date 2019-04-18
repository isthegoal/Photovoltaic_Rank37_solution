import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
import sklearn
from keras.callbacks import Callback
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler  # 这是标准化处理的语句，很方便，里面有标准化和反标准化。。
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
import os

def smape_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False

def min_max_normalize(data):
    # 归一化       数据的归一化计算，这样计算之后结果能更加适合非树模型，  但是进行归一化之后怎么反归一化得看下
    #数据量大，标准化慢
    df=data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # 做简单的平滑,试试效果如何
    return df

def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)

def NN_Model():
    '''
        训练nn模型，并提取，倒数第二层的特征，特征的提取方法可参照：https://blog.csdn.net/hahajinbu/article/details/77982721
        进行最后一层的抽取方法是， 先训练一个nn模型model，但是要提前给每层都赋好层命名，   之后再简历一个Model,输入是上一个模型
        训练所使用到的数据，输出是上一个model的指定层名，最为输出，然后使用Model去做预测，得到输出那一层结果
        '''
    train = pd.read_csv( './data/do_feat_data_2/train.csv')
    test = pd.read_csv( './data/do_feat_data_2/test.csv')


    train.pop('板温**1/2')
    train.pop('现场温度**1/2')
    test.pop('板温**1/2')
    test.pop('现场温度**1/2')
    #print('ddd:',len(test.columns))
    train=train.interpolate()
    train=train[~train.isin([np.nan,np.inf,-np.inf]).any(1)]
    #train.replace(np.inf,999)
    y = train.pop('发电量')
    #test=test.interpolate()
    #test=test[~test.isin([np.nan,np.inf,-np.inf]).any(1)]

    # f['data'].value存放的是时间戳 上空间的流量数据
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train=scaler.fit_transform(train)
    #test = scaler.fit_transform(test)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(y.reshape(-1, 1))
    y=scaler.fit_transform(y.reshape(-1, 1))

    #print(pm25_data_all.head())
    #print('000000000000000000',pm25_data_all.columns.values.tolist())

    #pm25_data_nor = min_max_normalize(pm25_data_all)
    #print(pm25_data_nor.columns.values.tolist())

    train_x, valid_x, train_y, valid_y = train_test_split(train,y,
                                                        test_size=0.2, random_state=11)


    model = Sequential()
    model.add(Dense(activation='relu', units=800, input_dim=902))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=200, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))
    optimizer = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=optimizer)

    # mc = ModelCheckpoint(filepath="./model/weights-improvement-{epoch:02d}-{val_auc:.2f}.h5", monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=0)
    es = EarlyStopping(monitor='val_rmse', patience=10, verbose=1, mode='min')
    model.fit(x=train_x, y=train_y, batch_size=32, epochs=200,
              validation_data=(valid_x, valid_y), verbose=1, callbacks=[es])


    model_file = os.getcwd() +'/model_saver/nn_1.model'
    model.save_weights(model_file, overwrite=True)


    # 开始进行抽取
    dense1_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)

def use_nn_to_gbdt(train_air='pm25'):
    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=800, input_dim=902))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=200, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))

    ####################   加载网络模型，和加载全部数据，作前40W做初始化和提取  ###############
    train = pd.read_csv( './data/do_feat_data_2/train.csv')
    test = pd.read_csv( './data/do_feat_data_2/test.csv')
    train.pop('板温**1/2')
    train.pop('现场温度**1/2')
    test.pop('板温**1/2')
    test.pop('现场温度**1/2')

    train=train.interpolate()
    train=train[~train.isin([np.nan,np.inf,-np.inf]).any(1)]
    test=test.interpolate()
    test.replace(np.inf,999)
    test.replace(-np.inf, -999)
    test=test[~test.isin([np.nan,np.inf,-np.inf]).any(1)]
    y = train.pop('发电量')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train=pd.DataFrame(scaler.fit_transform(train))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(y.reshape(-1, 1))
    y=scaler.fit_transform(y.reshape(-1, 1))

    model_file = os.getcwd() +'/model_saver/nn_1.model'
    model.load_weights(model_file)

    dense2_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)
    dense2_output = dense2_layer_model.predict(train)

    print(pd.DataFrame(dense2_output).head)
    #拼接标签

    #################   3.提取出来的特征作为GBDT的输入，重新训练一个模型  ########################

    dense2_output=pd.DataFrame(dense2_output)
    gbm0 = GradientBoostingRegressor()

    gbm0.fit(dense2_output, y)

    model_file=os.getcwd() +'/model_saver/nn_gbdt.model'
    with open(model_file, 'wb') as fout:
        pickle.dump(gbm0, fout)
    # test_Y1 = gbm0.predict(test_X)
    # score = get_score(test_Y1, test_Y)

def gbdt_nn_predict(train_air='pm25'):
    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=800, input_dim=902))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=200, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))
    ###########################################################
    ####################   加载网络模型，和加载全部数据，作前40W做初始化和提取  ###############
    train = pd.read_csv('./data/do_feat_data_2/train.csv')
    test = pd.read_csv('./data/do_feat_data_2/test.csv')
    train.pop('板温**1/2')
    train.pop('现场温度**1/2')
    test.pop('板温**1/2')
    test.pop('现场温度**1/2')

    test.replace(np.inf,np.nan,inplace=True)
    test.replace({-np.inf:np.nan},inplace=True)
    test.replace({np.nan: 1}, inplace=True)
    #使用前向填充方法
    test.fillna(method='pad')
    train = train.interpolate()
    train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
    test = test.interpolate()
    #test.to_csv(os.getcwd() +'/model_saver/show.csv')
    #print(':::::::::::',np.isinf(test))
    if np.isinf(test['Unnamed: 0/风速'][8]):
        print(':::::', test['Unnamed: 0*电压C'][1])
    test.to_csv(os.getcwd() +'/model_saver/show.csv')
    #test.replace(np.nan, 0)
    #test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
    y = train.pop('发电量')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    test = pd.DataFrame(scaler.fit_transform(test))
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler1.fit(y.reshape(-1, 1))
    y = scaler1.fit_transform(y.reshape(-1, 1))

    ###########################  用神经网络作提取  #########################
    model_file = os.getcwd() +'/model_saver/nn_1.model'
    model.load_weights(model_file)

    dense2_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)
    dense2_output = dense2_layer_model.predict(test)
    dense2_output = pd.DataFrame(dense2_output)

    print('dense2_output:',dense2_output)

    model_path=os.getcwd() +'/model_saver/nn_gbdt.model'
    model = pickle.load(open(model_path, 'rb'))

    yy=model.predict(dense2_output)
    yy = scaler1.inverse_transform(yy.reshape(-1, 1))
    print(',,,,,:',yy)

    lgbm_submission = pd.read_csv('./data/base_data/submit_example.csv',header=None)
    lgbm_submission.columns=['A','B']
    res = pd.DataFrame()
    res['A'] = list(lgbm_submission['A'])
    res['B'] = list(yy)
    #print(res[res['RST'] == 1].shape)

    ####################################    提交和线下结果展示   #####################################
    res.to_csv(os.getcwd() +'/submit/%s_%s.csv'% (str(12), str(21)),
    index=False, header=None)



if __name__=='__main__':
    print('************    1.训练nn模型     *************')
    NN_Model()
    print('************    2.训练gbdt模型     *************')
    use_nn_to_gbdt()
    print('************    3.对test集进行预测     *************')
    gbdt_nn_predict()