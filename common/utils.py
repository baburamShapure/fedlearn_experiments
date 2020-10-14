from torch.utils.data import DataLoader
import argparse

def getDataLoader(train, test, train_batch = 512, test_batch = 1000):
    trainLoad = DataLoader(train, batch_size=train_batch, shuffle=True)
    testLoad = DataLoader(test, batch_size= test_batch)

    return trainLoad, testLoad

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for Gradient Descent")
    args = parser.parse_args()
    return args

