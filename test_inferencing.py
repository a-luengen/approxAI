from net.ResNet import resnet50
import os
import torch
import torchvision
from utils import getDataLoaders, classes

load_model_name = 'resnet_50'
load_model_time = '12-09-05-12-Sep-2019'

if __name__ == '__main__':
    net = resnet50(False)
    #load_path = os.path.join('./checkpoints', load_model_name, load_model_time)
    #print('Loading: ' + load_path)
    #net.load_state_dict(torch.load(load_path))
    data_per_tensor = 1

    trainLoader, testLoader = getDataLoaders(data_per_tensor, data_per_tensor)
    net.eval()

    # test the network
    dataIter = iter(testLoader)
    images, labels = dataIter.next()

    #imgShow(torchvision.utils.make_grid(images))
    print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(data_per_tensor)))

    outputs = net(images)
    #print(outputs)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    print('Predicted:  ', '|'.join('%5s' % classes[predicted[k]] for k in range(data_per_tensor)))

