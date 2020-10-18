import torch.nn as nn 
import torch.nn.functional as F 

class FFN(nn.Module): 
    def __init__(self, input_dim, num_classes): 
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
    
    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = self.bn1(out)
        out = nn.Dropout()(out)
        out = F.relu(self.linear2(out))
        out = nn.Dropout()(out)
        out = self.out(out)
        return out 

class FFNBig(nn.Module): 
    def __init__(self, input_dim, num_classes): 
        super(FFNBig, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_classes)
    
    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = nn.Dropout()(out)
        # out = nn.BatchNorm1d()(out)
        out = F.relu(self.linear2(out))
        out = nn.Dropout()(out)
        out = F.relu(self.linear3(out))
        out = nn.Dropout()(out)
        out = F.relu(self.linear4(out))
        out = nn.Dropout()(out)
        out = self.out(out)
        return out 
