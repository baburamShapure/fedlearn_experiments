from torch.utils.data import DataLoader
import argparse
import tqdm 
import sklearn.metrics as metrics 
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type = str, default='hhar', help="dataset to use for training")
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for Gradient Descent")
    args = parser.parse_args()
    return args


def getDataLoader(train, test, train_batch = 512, test_batch = 1000):
    trainLoad = DataLoader(train, batch_size=train_batch, shuffle=True)
    testLoad = DataLoader(test, batch_size= test_batch)

    return trainLoad, testLoad

def train_central_one_epoch(model, trainloader, testloader, optimizer, loss):
    for i, (x, y) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        yhat = model(x)
        batch_loss = loss(yhat, y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    # evaluate on test dataset. 
    acc = get_accuracy(model, testloader)
    return acc     

def get_accuracy(model, loader):
    """given a dataloader object, 
    get predictions of model on each minibatch. 
    outputs a list of minibatch losses. 
    Use these to plot standard errors. 
    """ 
    acc_list = []
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            oos_scores = model(x)
        oos_preds = torch.max(oos_scores, 1)[1].numpy()  
        # won't work with a gpu.   
        acc = metrics.accuracy_score(y.cpu().numpy(), oos_preds)
        acc_list.append(acc)
    return acc_list



