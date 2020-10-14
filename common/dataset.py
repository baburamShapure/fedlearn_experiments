import os 
import pandas as pd 
import torch 
from torch.utils.data import Dataset
import numpy as np 

def prepareData(datadir = 'data/FL_data'):
    # read train data from each folder. 
    trainlist = [] 
    testlist = [] 
    for each_folder in os.listdir(datadir): 
        test, train = os.listdir(os.path.join(datadir, each_folder))
        trainlist.append(pd.read_csv(os.path.join(datadir, each_folder, train)))
        testlist.append(pd.read_csv(os.path.join(datadir, each_folder, test)))
    trainData = pd.concat(trainlist)
    testData = pd.concat(testlist)
    return trainData, testData


class HARData(Dataset): 
    def __init__(self, data, mode="train"):
        self.data = data

    # data set size
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        floats = row.drop(["Activity"]).values.astype(np.float32)
        data = torch.tensor(floats)

        y_val = self.data.iloc[idx]["Activity"]
        y = torch.tensor(y_val, dtype=torch.long)
        return data, y

