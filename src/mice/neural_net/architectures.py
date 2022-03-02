import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MiceConv(nn.Module):
    def __init__(self, input_size=576):
        super().__init__()

        self.layer1 =  nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
                                     nn.ReLU())
        

        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear( int(input_size/2),  1)
        
        print('Finished init')

    def forward(self, data):
        output = self.layer1(data)
        output = self.layer2(output)

        output = self.drop_out(output)
        output = output.reshape(output.size(0), -1)

        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

class Net(nn.Module):
    '''
    The linear architecture of the neural net
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 2 * 2 * 2)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x


class Model(nn.Module):
    '''
    The semi fully conventional architecture of the neural net
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(2,2), stride=(1,1), padding=1,)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1, )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1, )
        self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global1(x)
        x = self.avgpool(x)
        x = x.view(-1, 2 * 2 * 2)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x


class Modely(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self, input_size):
        super(Modely, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1),)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), )
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout1 = nn.Dropout(p=0.2) # p – probability of an element to be zeroed. Default: 0.5
        self.batchnorm1 = nn.BatchNorm3d(num_features=16)
        self.batchnorm2 = nn.BatchNorm3d(num_features=32)
        self.batchnorm3 = nn.BatchNorm3d(num_features=64)
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        wide = torch.unsqueeze(x, 1) # torch.Size([32, 1, 1, 16, 16])
        x = self.conv1(F.relu(wide)) # torch.Size([32, 16, 1, 16, 16])
        x = self.conv2(F.relu(x)) # torch.Size([32, 1, 1, 16, 16])
        wide1, x1 = wide, x
        x2 = self.conv1(F.relu(x1 + wide1)) # torch.Size([32, 16, 1, 16, 16])
        x2 = self.conv2(F.relu(x2)) # torch.Size([32, 1, 1, 1, 1])
        x3 = self.conv1(F.relu(x2 + wide1)) # torch.Size([32, 16, 1, 16, 16])
        x3 = self.conv2(F.relu(x3)) # torch.Size([32, 1, 1, 16, 16])
        x3 = x3.squeeze(axis=1) # torch.Size([32, 1, 16, 16])
        x4 = x3.view(x3.shape[0], -1) # torch.Size([32, 256])
        x4 = self.fc1(x4) # torch.Size([32, 1])
        x5 = self.avgpool3d(x3) # torch.Size([32, 1, 1, 1])
        x5 = x5.view(x5.shape[0], -1) # torch.Size([32, 1])
        return x4

class Sandnet(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self, input_size=576):
        super(Sandnet, self).__init__()


        self.layer1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1,3,3), stride=(1,1,1), padding=0,)
        self.layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(1,3,3), stride=(1,1,1), padding=0,)
        self.layer3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0,)
        self.drop_out = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear( int(input_size/2),  1)

    def forward(self, data):
        output = torch.unsqueeze(data, 1)
        output = F.relu(self.layer1(output))
        output = F.relu(self.layer2(output))
        output = self.layer3(output)
        output = self.drop_out(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output


class Sandnet3d(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self, input_size=576):
        super(Sandnet3d, self).__init__()

        self.input_size = input_size
        self.layer1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=2, stride=1, padding=0,)
        self.layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0,)
        self.layer3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0,)
        self.drop_out = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_size , int(input_size/2))
        self.fc2 = nn.Linear(int(input_size/2), 1)

    def forward(self, data):
        output = F.relu(self.layer1(data))
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

class Sandnet2d(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self, input_size=576):
        super(Sandnet2d, self).__init__()

        self.input_size = input_size
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1, padding=0,)
        self.layer2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0,)
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,)
        self.drop_out = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_size , int(input_size/2))
        self.fc2 = nn.Linear(int(input_size/2), 1)

    def forward(self, data):
        output = F.relu(self.layer1(data))
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

class Sandnet3d_emb(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self, input_size=576, sizes=(10,10,10)):
        super(Sandnet3d_emb, self).__init__()

        size = 8 * (sizes[0]-5) * (sizes[1]-5) * (sizes[2]-5)
        self.input_size = input_size
        self.layer1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=2, stride=1, padding=0,)
        self.layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0,)
        self.layer3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0,)
        self.layer4 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=0,)
        self.layer5 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=2, stride=1, padding=0,)
        self.layer6 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0,)
        self.drop_out = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(size , int(size/2))
        self.fc2 = nn.Linear(int(size/2), 1)

    def forward(self, data):
        output = F.relu(self.layer1(data))
        output = F.relu(self.layer2(output))
        output = F.relu(self.layer3(output))
        output = F.relu(self.layer4(output))
        output = F.relu(self.layer5(output))
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output