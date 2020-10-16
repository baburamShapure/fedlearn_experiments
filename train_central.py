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

def train_central_one_epoch(model, trainloader, testdata, optimizer, loss):
    for i, (x, y) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        yhat = model(x)
        batch_loss = loss(yhat, y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    # evaluate on test dataset. 

    with torch.no_grad():
        oos_scores = model(torch.FloatTensor(testdata.iloc[:, :-1].values))
    oos_preds = torch.max(oos_scores, 1)[1].numpy()    

    acc = metrics.accuracy_score(testData['Activity'], oos_preds)
    return acc     

if __name__ == '__main__': 

    runtimestamp = dt.datetime.strftime(dt.datetime.today(), 
                            '%Y%m%d_%H%M%S')
    
    logfile = os.path.join('logs', runtimestamp + '.log')
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    #Creating an object 
    logger=logging.getLogger()   
    #Setting the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 
    args = args_parser()
    logger.info('Experiment started')
    logger.info('Hyper-parameters: {0}'.format(args))
    # read data. 
    trainData, testData = prepareData()
    trainHAR = HARData(trainData)
    testHAR = HARData(testData)

    trainLoad, testLoad = getDataLoader(trainHAR, testHAR, args.batch_size)
    model = FFN(26, 10)
    logger.info('Model architecture: {0}'.format(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr )
    loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        acc = train_central_one_epoch(model= model, 
                                trainloader=trainLoad,
                                testdata=testData, 
                                optimizer=optimizer,
                                loss = loss
                                )
        logger.info('Epoch : {0}, Validation Accuracy: {1}'.format(epoch, acc))
    
    model_out = os.path.join('models', runtimestamp)
    torch.save(model, model_out)


