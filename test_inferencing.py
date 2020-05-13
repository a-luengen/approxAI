from net.ResNet import resnet50
from sklearn import metrics
import os
import torch
import torchvision
from utils import getDataLoaders, classes

load_model_name = 'resnet_50'
load_model_time = '12-09-05-12-Sep-2019'
load_model_state = '123-best.pth'

def printStats(ground_truth, predicted):
    print("Confussion matrix:")
    print(metrics.confusion_matrix(ground_truth, predicted))
    print("Recall and precision:")
    print(metrics.classification_report(ground_truth, predicted, digits=3))

def loadAndEvalWithStateDict():
    net = resnet50(True)
    load_path = os.path.join('./state_dicts', load_model_name + "-" + load_model_state)
    print('Loading: ',load_path)
    net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    print('Loading Test Data..')
    data_per_tensor = 100
    _, testLoader = getDataLoaders(data_per_tensor, data_per_tensor)
    net.eval()
    print('Loaded testData with {} testImages and {} images per tensor.'.format(len(testLoader.dataset), data_per_tensor))

    pred, grndT = [], []
    for i, (images, labels) in enumerate(iter(testLoader)):
        print("Evaluating: ", i, "-th iteration")
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        pred = pred + [classes[predicted[k]] for k in range(data_per_tensor)]
        grndT = grndT + [classes[labels[j]] for j in range(data_per_tensor)]
        #if i == 1:
        #    break

    printStats(grndT, pred)

if __name__ == '__main__':

    loadAndEvalWithStateDict()

    quit()

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

