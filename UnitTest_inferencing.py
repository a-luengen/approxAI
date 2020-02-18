import unittest

from net.ResNet import resnet50
import os
import torch
import torchvision
from utils import getDataLoaders, classes



class Test(unittest.TestCase):

    load_model_name = "resnet_50"
    load_model_time = '12-09-05-12-Sep-2019'

    def test_0_createResNet50_and_inference_input_without_exception(self):
        net = resnet50(False)

        batch_size = 1

        _, testLoader = getDataLoaders(batch_size, batch_size)

        net.eval()

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        print(predicted)

    def test_1_createResNet50_with_convolution_and_inference_input_without_exception(self):
        net = resnet50(True)

        batch_size = 1
        _, testLoader = getDataLoaders(batch_size, batch_size)

        net.eval()

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print(predicted)

if __name__ == "__main__":
    unittest.main()