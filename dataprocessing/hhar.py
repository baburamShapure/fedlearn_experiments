"""
Dataset specific adaptation pipeline 
for each data source. 

Data for Fedlearn: 

id: 
    | agent-1 
            | train.csv
            | test.csv
    |agent-2
            |train.csv
            |test.csv 
        .
        .
        .
This script adapts the heterogeneity human activity 
recognition dataset into the federated learning 
folder structure. 


For the first version, we will not follow the paper. 
This version rounds off the datetime to the nearest deci-second and takes average, 
standard_deviation, min and max over the x, y & z measurements. 

This has a similar effect of a 100 hz sampling rate. 
"""

import pandas as pd 
import os 
import numpy as np 
import datetime as dt
import gc
import string
import tqdm 
from sklearn.model_selection import train_test_split

def convert_to_decisecond(df): 
    """
    round off timestamp to deci-second 
    to make the sampling frequency to 100 Hz. 
    """
    df['arrivalTimeDttm'] = pd.to_datetime(df['Arrival_Time'], unit ='ms')
    # strip away milliseconds. 
    df['arrivalTimeDttm_rounded'] = df['arrivalTimeDttm'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    df['arrivalTimeDttm_rounded'] = df['arrivalTimeDttm'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    return pd.to_datetime(df['arrivalTimeDttm_rounded']) + pd.to_timedelta(round(df['arrivalTimeDttm'].dt.microsecond /10000000, 2) , 's')  

def aggregate(df, prefix):
    summary = df[['gt', 'arrivalDttm_rounded', 'x', 'y', 'z']].groupby(['gt', 'arrivalDttm_rounded']).agg([np.mean, np.std, min, max])
    summary.columns = summary.columns.to_flat_index()
    summary =summary.reset_index()
    summary.columns = [i  if len(i) == 0 or type(i)==str  else  '_'.join([prefix] + list(i)) for i in list(summary.columns.to_flat_index())]
    return summary

#TODO: Label encoding. 
CLASSMAPS = {'bike':0, 'sit': 1, 
                'stairsup': 2, 'stairsdown':3,
                 'stand':4, 'walk': 5}


HHAR_DATADIR = 'rawdata/HHAR/'
HHAR_OUTDIR = 'fl_data/HHAR'
PROP_TRAIN = 0.8
filesToRead = [os.path.join(HHAR_DATADIR, i) for i in os.listdir(HHAR_DATADIR) if i[-3:] == 'csv']
dataList = [pd.read_csv(each_file) for each_file in filesToRead]


USERS = list(string.ascii_letters[:9])
PREFIX = ['phone_acc', 'phone_gyro', 'watch_acc', 'watch_gyro']

for each_user in USERS: 
        print('Preparing data for User {0}'.format(each_user))
        joined = None 
        i = 0
        for each_data in tqdm.tqdm(dataList): 
                each_user_data = each_data[each_data['User'] == each_user]
                each_user_data['arrivalDttm_rounded'] = convert_to_decisecond(each_user_data)
                each_user_data = aggregate(each_user_data, prefix = PREFIX[i])
                i += 1
                print(each_user_data.shape)
                if joined is None: 
                        joined = each_user_data
                else: 
                        joined = joined.merge(each_user_data)

        print('Final Data Shape:  Num rows: {0}, Num cols: {1}'.format(joined.shape[0], joined.shape[1]))
        
        if each_user not in os.listdir(HHAR_OUTDIR): 
                os.mkdir(os.path.join(HHAR_OUTDIR, each_user))

        joined['response'] = joined['gt'].map(CLASSMAPS)
        joined = joined.drop('gt', axis  = 1)
        # shuffle the data first to break correlation. 
        joined = joined.sample(frac = 1)
        # train test split. 
        train, test = train_test_split(joined, test_size= 1 - PROP_TRAIN)

        # write. 
        train.to_csv(os.path.join(HHAR_OUTDIR, each_user,  'train.csv'), index = None)
        test.to_csv(os.path.join(HHAR_OUTDIR, each_user,  'test.csv'), index = None)
        


