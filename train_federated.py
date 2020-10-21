import os 
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import gc
from torch.utils.data import Dataset, DataLoader
import tqdm
import sklearn.metrics as metrics
from common.dataset import *
from common.models import * 
from common.utils import * 
import logging 
import argparse
import datetime as dt 
import random 
import copy

# len(keys)*len(weights) - can get expensive
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg


DATADIR = 'fl_data/HHAR'
FL_AGENTS = os.listdir(DATADIR)
FL_AGENTS

FL_SAMPLE = 0.4
global_model = FFN(48, 10)

random.sample(FL_AGENTS, 4)

LOCAL_MODELS = {}
for each_round in tqdm.tqdm(range(10)): 
    agents_to_train = random.sample(FL_AGENTS, k= int(FL_SAMPLE * len(FL_AGENTS)))
    model_list = []
    for each_agent in agents_to_train: 
        # read the data. 
        test, train = [pd.read_csv(os.path.join(DATADIR, each_agent, i)) for i in os.listdir(os.path.join(DATADIR, each_agent))]
        train = train.fillna(0)
        test = test.fillna(0)
        trainData, testData = HARData(train), HARData(test)
        trainLoader, testLoader = getDataLoader(trainData, testData)
        loss = nn.CrossEntropyLoss()
        model = copy.deepcopy(global_model)
        optimizer = optim.Adam(model.parameters())
        model.train()
        for epoch in range(10): 
            # train each epoch. 
            for i, (x, y) in enumerate(trainLoader): 
                yhat = model(x)
                batch_loss = loss(yhat, y)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
        print('Round: {0}, Agent: {1}'.format(each_round, each_agent))
        print(get_accuracy(model, testLoader))     
        LOCAL_MODELS[each_agent] = model # each agents gets to retain its local model.        
        model_list.append(model.state_dict())
    # average weight at end of round. 
    avg_weights = average_weights(model_list)
    global_model.load_state_dict(avg_weights)

