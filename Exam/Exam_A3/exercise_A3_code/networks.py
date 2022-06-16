import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10) # Creates a complete linear layer with 784 input features and 10 output features.
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # Used to flatten the 28x28 images to 784-dimensional vectors
        return self.layer(x)


class MLPModel(nn.Module):
    def __init__(self, outputNumberHiddenLayer=300):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(784,outputNumberHiddenLayer)
        self.linear2 = nn.Linear(outputNumberHiddenLayer,10)

    def forward(self, x):
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)



class CNNModel(nn.Module):
    def __init__(self, outputNumberHiddenLayer1=8, outputNumberHiddenLayer2=16, outputNumberHiddenLayerLinear = 300):
        super(CNNModel, self).__init__()
        self.kernelSize = 2
        convolutionKernelSize1 = 11    #28-->28-11+1=18
        convolutionKernelSize2 = 11    #18-11+1=8
        self.conv1 = nn.Conv2d(1, outputNumberHiddenLayer1, convolutionKernelSize1)
        self.conv2 = nn.Conv2d(outputNumberHiddenLayer1, outputNumberHiddenLayer2, convolutionKernelSize2)
        
        inputLinear1 = 4*4*outputNumberHiddenLayer2     #max pool: 8/2-->4
        self.linear1 = nn.Linear(inputLinear1,outputNumberHiddenLayerLinear)
        self.linear2 = nn.Linear(outputNumberHiddenLayerLinear,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        maxPool = nn.MaxPool2d(self.kernelSize)
        x = maxPool(x)
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)



class CNNModel2(nn.Module):
    def __init__(self, outputNumberHiddenLayer1=8, outputNumberHiddenLayer2=16, outputNumberHiddenLayer3=16, outputNumberHiddenLayerLinear = 300):
        super(CNNModel2, self).__init__()
        self.kernelSize = 2
        convolutionKernelSize1 = 9    #28-->28-9+1=20
        convolutionKernelSize2 = 9    #20-9+1=12
        convolutionKernelSize3 = 5    #12-5+1=8
        self.conv1 = nn.Conv2d(1,                        outputNumberHiddenLayer1, convolutionKernelSize1)
        self.conv2 = nn.Conv2d(outputNumberHiddenLayer1, outputNumberHiddenLayer2, convolutionKernelSize2)
        self.conv3 = nn.Conv2d(outputNumberHiddenLayer2, outputNumberHiddenLayer3, convolutionKernelSize3)
        
        inputLinear1 = 4*4*outputNumberHiddenLayer3     #max pool: 8/2-->4
        self.linear1 = nn.Linear(inputLinear1,outputNumberHiddenLayerLinear)
        self.linear2 = nn.Linear(outputNumberHiddenLayerLinear,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        maxPool = nn.MaxPool2d(self.kernelSize)
        x = maxPool(x)
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class CNNModel3(nn.Module):
    def __init__(self, outputNumberHiddenLayer1=8, outputNumberHiddenLayer2=16, outputNumberHiddenLayer3=16, outputNumberHiddenLayer4=16, outputNumberHiddenLayerLinear = 300):
        super(CNNModel3, self).__init__()
        self.kernelSize = 2
        convolutionKernelSize1 = 8    #28-->28-8+1=21
        convolutionKernelSize2 = 8    #21-8+1=14
        convolutionKernelSize3 = 5    #14-5+1=10
        convolutionKernelSize4 = 3    #10-3+1=8
        self.conv1 = nn.Conv2d(1, outputNumberHiddenLayer1, convolutionKernelSize1)
        self.conv2 = nn.Conv2d(outputNumberHiddenLayer1, outputNumberHiddenLayer2, convolutionKernelSize2)
        self.conv3 = nn.Conv2d(outputNumberHiddenLayer2, outputNumberHiddenLayer3, convolutionKernelSize3)
        self.conv4 = nn.Conv2d(outputNumberHiddenLayer3, outputNumberHiddenLayer4, convolutionKernelSize4)
        
        inputLinear1 = 4*4*outputNumberHiddenLayer4     #max pool: 8/2-->4
        self.linear1 = nn.Linear(inputLinear1,outputNumberHiddenLayerLinear)
        self.linear2 = nn.Linear(outputNumberHiddenLayerLinear,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        maxPool = nn.MaxPool2d(self.kernelSize)
        x = maxPool(x)
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)






class CNNModel_dropout(nn.Module):
    def __init__(self, outputNumberHiddenLayer1=8, outputNumberHiddenLayer2=16, p1=0.5, p2=0.5, outputNumberHiddenLayerLinear = 300):
        super(CNNModel_dropout, self).__init__()
        self.kernelSize = 2
        convolutionKernelSize1 = 11    #28-->28-11+1=18
        convolutionKernelSize2 = 11    #18-11+1=8
        self.conv1 = nn.Conv2d(1, outputNumberHiddenLayer1, convolutionKernelSize1)
        self.drop_layer1 = nn.Dropout(p=p1)
        self.conv2 = nn.Conv2d(outputNumberHiddenLayer1, outputNumberHiddenLayer2, convolutionKernelSize2)
        
        inputLinear1 = 4*4*outputNumberHiddenLayer2     #max pool: 8/2-->4
        self.linear1 = nn.Linear(inputLinear1,outputNumberHiddenLayerLinear)
        self.drop_layer2 = nn.Dropout(p=p2)
        self.linear2 = nn.Linear(outputNumberHiddenLayerLinear,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop_layer1(x)
        x = F.relu(self.conv2(x))
        maxPool = nn.MaxPool2d(self.kernelSize)
        x = maxPool(x)
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.drop_layer2(x)
        return self.linear2(x)


