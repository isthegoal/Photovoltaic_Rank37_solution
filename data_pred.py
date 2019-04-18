import pandas as pd
import numpy as np
from pandas import DataFrame as DF


if __name__=='__main__':
    print('开始预处理')

    train = pd.read_csv('./data/base_data/public.train.csv')
    test = pd.read_csv('./data/base_data/public.test.csv')
    # train.loc[train['现场温度'] <-50, '现场温度'] = np.nan
    # train.loc[train['风速'] >10, '风速'] = np.nan
    # train.loc[train['风向'] >360, '风速'] = np.nan
    # train.loc[train['电压A'] >800, '电压A'] = np.nan
    # train.loc[train['电压B'] >800, '电压B'] = np.nan
    # train.loc[train['电压C'] >800, '电压C'] = np.nan
    # train.loc[train['电流B'] >20, '电流B'] = np.nan
    # train.loc[train['电流C'] >20, '电流C'] = np.nan
    # train.loc[train['功率A'] >10000, '功率A'] = np.nan
    # train.loc[train['功率B'] >10000, '功率B'] = np.nan
    # train.loc[train['功率C'] >10000, '功率C'] = np.nan
    # train.loc[train['平均功率'] >10000, '平均功率'] = np.nan
    #
    #
    # test.loc[train['现场温度'] <-50, '现场温度'] = np.nan
    # test.loc[train['风速'] >10, '风速'] = np.nan
    # test.loc[train['风向'] >360, '风速'] = np.nan
    # test.loc[train['电压A'] >800, '电压A'] = np.nan
    # test.loc[train['电压B'] >800, '电压B'] = np.nan
    # test.loc[train['电压C'] >800, '电压C'] = np.nan
    # test.loc[train['电流B'] >20, '电流B'] = np.nan
    # test.loc[train['电流C'] >20, '电流C'] = np.nan
    # test.loc[train['功率A'] >10000, '功率A'] = np.nan
    # test.loc[train['功率B'] >10000, '功率B'] = np.nan
    # test.loc[train['功率C'] >10000, '功率C'] = np.nan
    # test.loc[train['平均功率'] >10000, '平均功率'] = np.nan
    #
    #train=train.interpolate()
    # test=test.interpolate()

    print('去除异常值')


    train.to_csv('./data/do_data_pred/train.csv')
    test.to_csv('./data/do_data_pred/test.csv')