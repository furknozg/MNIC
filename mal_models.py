import torch
import torch.nn as nn


class MalwareIdentifier(nn.Module):
    # Fully connected 3 class classifier (assumed output properties [benign,malware,ood/suspicious])
    def __init__(self, input_size, hidden_size, output_size=3):
        super(MalwareIdentifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lin1_bn = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))

        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/2))

        self.fc4 = nn.Linear(int(hidden_size/2), output_size)

        self.smo = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.lin1_bn(self.fc1(x)))
        x = self.fc2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.smo(x)
        

        return x
    


class MalwareClassifier(nn.Module):
    #Fully connected model (assumed input will be from known malware data)
    def __init__(self, input_size, hidden_size, num_classes):
        super(MalwareClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.BatchNorm1d(int(hidden_size/2))

        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.bn3 = nn.BatchNorm1d(int(hidden_size/4))

        
        self.fc4 = nn.Linear(int(hidden_size/4), num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)

        self.smo = nn.LogSoftmax(dim=1)


    def forward(self, x):
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = self.bn3(self.fc3(out))
        out = self.bn4(self.fc4(out))
        out = self.smo(out)
        return out