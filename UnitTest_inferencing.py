import unittest

import time
import os
from net.ResNet import resnet50
import torch
import torchvision
import random
from utils import getDataLoaders, classes



class Test(unittest.TestCase):

    load_model_name = "resnet_50"
    load_model_time = '12-09-05-12-Sep-2019'

    def equalizeWeights(self, torch_net, ocl_net):
        # weights have to be equal
        PATH = "checkpoints/test.dict"
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(ocl_net.state_dict(), PATH)
        torch_net.load_state_dict(torch.load(PATH))

    #@unittest.skip("")
    def test_0_createResNet50_and_inference_input_without_exception(self):
        net = resnet50(False)

        batch_size = 1

        _, testLoader = getDataLoaders(batch_size, batch_size)

        net.eval()

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        #print(predicted)

    #@unittest.skip("")
    def test_1_createResNet50_with_convolution_and_inference_input_without_exception(self):
        net = resnet50(True)

        batch_size = 1
        _, testLoader = getDataLoaders(batch_size, batch_size)

        net.eval()

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        #print("Run inferencing...")
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        
        print(predicted)
        print("Predicted:    ", classes[predicted[0]])
        print("Ground Truth: ", classes[labels[0]])

    #@unittest.skip("")
    def test_2_getDimensionsOfInputDataForRandomBatchSize_testForDimensionality(self):
        batch_size = random.randrange(1, 15)
        _, testLoader = getDataLoaders(batch_size, batch_size)
        
        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        self.assertEqual(images.size()[0], batch_size)
        self.assertEqual(images.size()[1], 3)
        self.assertEqual(images.size()[2], 32)
        self.assertEqual(images.size()[3], 32)

    def test_3_testResNet50WithOCLagainstPytorchImplementation(self):

        netOcl = resnet50(True)
        netpy = resnet50(False)
        self.equalizeWeights(netpy, netOcl)

        batch_size = 100
        _, testLoader = getDataLoaders(batch_size, batch_size)

        netOcl.eval()
        netpy.eval()

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        start = time.time()
        outputs_ocl = netOcl(images)
        print("      Ocl Inference Time [sec]: ", time.time() - start)

        start = time.time()
        outputs_py = netpy(images)
        print("Pytorch-Inferencing Time [sec]: ", time.time() - start)

        _, predicted_ocl = torch.max(outputs_ocl, 1)
        _, predicted_py = torch.max(outputs_py, 1)
        
        for (pred_ocl, pred_py) in zip(predicted_ocl, predicted_py):
            #print("ocl-predict: ", classes[pred_ocl])
            #print("py-predict:  ", classes[pred_py])

            self.assertTrue(torch.equal(pred_ocl, pred_py))
            self.assertEqual(classes[pred_py], classes[pred_ocl], 
                "OpenCL and reference PyTorch implementation predictions are not equal: torch={} - ocl={}."
                    .format(classes[pred_py], classes[pred_ocl])
            )
    @unittest.skip("")
    def test_4_inferenceOCL_with_batchSize100_againstPyTorch(self):
        netOcl = resnet50(True)
        netpy = resnet50(False)

        batch_size = 100


if __name__ == "__main__":
    unittest.main()