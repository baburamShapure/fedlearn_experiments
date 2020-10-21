import os 
import pandas as pd 
import torch 
from torch.utils.data import Dataset
import numpy as np 

def prepareData(datadir):
    """read train data from each folder. 
    traindata is a single dataframe while testdata is a dict of dataframe
    """
    trainlist = [] 
    testlist = {} 
    for each_folder in os.listdir(datadir): 
        test, train = os.listdir(os.path.join(datadir, each_folder))
        trainlist.append(pd.read_csv(os.path.join(datadir, each_folder, train)))
        tmp = pd.read_csv(os.path.join(datadir, each_folder, test))
        testlist[each_folder] = tmp.fillna(0)
    trainData = pd.concat(trainlist)
    trainData = trainData.fillna(0)
   
    return trainData, testlist


class HARData(Dataset): 
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

