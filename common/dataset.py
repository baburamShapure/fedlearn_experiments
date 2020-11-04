import os 
import pandas as pd 
import torch 
from torch.utils.data import Dataset
import numpy as np 

def prepareData(datadir, mode = 'central'):
    """read train data from each folder. 
    if mode is central   traindata is a single dataframe 
    if mode is federated, traindata is dict of dataframes. 
    testdata is always dict of dataframe
    """

    trainlist = {}
    testlist = {} 
    
    for each_folder in os.listdir(datadir): 
        test, train = os.listdir(os.path.join(datadir, each_folder))
        tmp = pd.read_csv(os.path.join(datadir, each_folder, train))
        trainlist[each_folder] = tmp.fillna(0)

        tmp = pd.read_csv(os.path.join(datadir, each_folder, test))
        testlist[each_folder] = tmp.fillna(0)
    
    if mode == 'central': 
        trainData = pd.concat(trainlist)
        trainData = trainData.fillna(0)
        return trainData, testlist
    elif mode == 'federated': 
        return trainlist, testlist 



class Data(Dataset): 
    """
    Assumes the features are present in column 1 onwards. 
    The response is in the column named response. 
    """
    def __init__(self, data, mode="train"):
        self.data = data

    # data set size
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx, 1:]
        floats = row.drop(["response"]).values.astype(np.float32)
        data = torch.tensor(floats)

        y_val = self.data.iloc[idx]["response"]
        y = torch.tensor(y_val, dtype=torch.long)
        return data, y

def to_dataset(data):
    """
    given a dict of datasets, convert to the 
    dataset class using Data. 
    """
    print(type(data) == 'dict')
    if type(data) == 'dict':
        out = {}
        for k in data.keys():
            out[k] = Data(data[k])
    else:
        out = Data(data)
    return out 

    