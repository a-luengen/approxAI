import unittest

from net.ResNet import resnet50
import torch
import torchvision
import random
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

        print("Run inferencing...")
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        
        print(predicted)
        print("Predicted:    ", classes[predicted[0]])
        print("Ground Truth: ", classes[labels[0]])

    def test_2_getDimensionsOfInputDataForRandomBatchSize_testForDimensionality(self):
        batch_size = random.randrange(1, 15)
        _, testLoader = getDataLoaders(batch_size, batch_size)
        
        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        self.assertEqual(images.size()[0], batch_size)
        self.assertEqual(images.size()[1], 3)
        self.assertEqual(images.size()[2], 32)
        self.assertEqual(images.size()[3], 32)

if __name__ == "__main__":
    unittest.main()