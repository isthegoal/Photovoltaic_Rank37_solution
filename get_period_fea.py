import pandas as pd
import numpy as np
from pandas import DataFrame as DF



if __name__=='__main__':
    train = pd.read_csv( './data/do_data_pred/train.csv')
    test = pd.read_csv( './data/do_data_pred/test.csv')
    sun_qiangdu=train['光照强度']
    #print(sun_qiangdu)
    start_cal_period=0
    period_index=[]
    while(start_cal_period+120<len(list(sun_qiangdu))):
        min_val=np.min(train['光照强度'][start_cal_period+60:start_cal_period+120])
        for i in range(start_cal_period+60,start_cal_period+120):

            if(sun_qiangdu[i]==min_val):
                start_cal_period=i
                period_index.append(i)
                break

    print('周期点为：：：',period_index)


        #cal_min(train['光照强度'][index+60:index+120],[i for i in range(index,index+120)])

