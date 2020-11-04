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
    if args.dataset == 'hhar' :
        datadir = r'fl_data\HHAR'
        input_dim = 48
        nclasses = 10
    else:
        #TODO other datasets here. Write better switch cases
        pass 

    trainData, testData  = prepareData(datadir, mode = 'central')
    train = to_dataset(trainData)
    test = to_dataset(testData)

    print(test)

    trainLoad, testLoad = getDataLoader(train, test, args.batch_size)

    model = FFN(input_dim, nclasses)
    logger.info('Model architecture: {0}'.format(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr )
    loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_one_epoch(model= model, 
                                trainloader=trainLoad,
                                optimizer=optimizer,
                                loss = loss)
        acc= evaluate(model, testLoad)
        
        logger.info('Epoch: {0}, {1}'.format(epoch, acc))
    
    model_out = os.path.join('models', runtimestamp)
    torch.save(model, model_out)


