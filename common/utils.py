from torch.utils.data import DataLoader
import argparse
import tqdm 
import sklearn.metrics as metrics 
import torch
import scipy as sp 

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type = str, default='hhar', help="dataset to use for training")
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for Gradient Descent")
    args = parser.parse_args()
    return args

def getDataLoader(train, train_batch = 512):
    trainLoad = DataLoader(train, batch_size=train_batch, shuffle=True)
    return trainLoad

def train_central_one_epoch(model, trainloader, optimizer, loss):
    model.train()
    for i, (x, y) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        # print('Num nans ', torch.sum(torch.isnan(x)))
        yhat = model(x)
        batch_loss = loss(yhat, y)
        # print('Batch loss: {0}'.format(batch_loss))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()


def get_accuracy(model, loader):
    """given a dataloader object, 
    get predictions of model on each minibatch. 
    outputs a list of minibatch losses. 
    Use these to plot standard errors. 
    """ 
    model.eval()
    acc_list = []
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            oos_scores = model(x)
        oos_preds = torch.max(oos_scores, 1)[1].numpy()  
        # won't work with a gpu.   
        acc = metrics.accuracy_score(y.numpy(), oos_preds)
        acc_list.append(str(acc))
    return ','.join(acc_list)


def evaluate(model, testLoader): 
    """
    model maybe central or federated. 
    testLoader is a dict of dataLoader objects. 
    Each key is an agent. 

    Returns list of accuracy for each agent. 
    """
    out = ''
    for each_key in testLoader.keys(): 
        out += " | Agent: {0}, Accuracy: {1}".format(each_key,
                         get_accuracy(model, loader=testLoader[each_key]))
    
    return out 





