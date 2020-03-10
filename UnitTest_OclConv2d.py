import unittest
import torch
from torch.nn import Conv2d
import numpy as np
import torchvision
import random
from utils import getDataLoaders, classes


# Unit testing for the Convolution implementation

from net.OCL_Conv2D import OCL_Conv2D

class Test(unittest.TestCase):

    batch_size = 1
    in_channels = 1
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 0
    padding_mode = 'zeros'
    dilation = 1
    bias = False

    def test_0_create_convolution_component_without_exception(self):
        oclComponent = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)
        self.assertTrue(oclComponent is not None)

    def test_1_convolution_returns_tensor_when_forwarding(self):
        tensor = torch.ones((1, 1, 10, 10), dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)
        result = convolution.forward(tensor)

        self.assertTrue(isinstance(result, torch.Tensor))
    
    def test_2_convolution_returns_tensor_with_correct_dimensions_on_forward(self):
        tensor = torch.ones((1, 1, 11, 11), dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)
        result = convolution.forward(tensor)

        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.out_channels)
        self.assertEqual(result.shape[2], tensor.shape[2] - 2)
        self.assertEqual(result.shape[3], tensor.shape[3] - 2)

    def test_3_convolution_with_stride_returns_correct_dimensions(self):
        #tensor = torch.ones(10, 10, 1, dtype=torch.float32)
        tensor = torch.ones((1, 1, 10, 10), dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3
        self.stride = 2

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, use_ocl=True)
        result = convolution.forward(tensor)
        
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.out_channels)
        self.assertEqual(result.shape[2], 4)
        self.assertEqual(result.shape[3], 4)

    def test_4_OCLconvolution2D_returns_numpy_array_as_result(self):
        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, use_ocl=True)
        test_input = np.ones((10, 10), dtype=np.float32)
        test_filter = np.ones((3, 3), dtype=np.float32)

        test_result = convolution.OCLconvolution2D(test_input, test_filter)
        self.assertIsNotNone(test_result)
        self.assertTrue(len(test_result.shape) == 2)

    def test_5_OCLconvolution2D_returns_dtype_float(self):
        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=1)
        test_input = np.ones((5,5), dtype=np.float32)
        test_filter = np.ones((3, 3), dtype=np.float32)

        test_result = convolution.OCLconvolution2D(test_input, test_filter)
        self.assertTrue(type(test_result) == type(test_input))
    
    def test_6_OCLconvolution2D_returns_correct_result_for_ones_convolution(self):
        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=1)
        test_input = np.ones((10, 10), dtype=np.float32)
        test_filter = np.zeros((3,3), dtype=np.float32)
        test_filter[1, 1] = 1.0

        test_result = convolution.OCLconvolution2D(test_input, test_filter)

        for row in test_result:
            for entry in row:
                self.assertEqual(entry, 1.0)

    def test_7_getNumpyOutputDimensions_returns_correct_dimensions(self):
        np_kernel = np.array((self.kernel_size, self.kernel_size))
        np_input = np.array((10, 10))
        output_dim = np.array([8, 8], dtype=np.int32)

        convolution_layer = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)

        result_dim = convolution_layer.getNumpyOutputDimensions(np_input, np_kernel)

        self.assertEqual(result_dim[0], output_dim[0])
        self.assertEqual(result_dim[1], output_dim[1])


    def test_8_performOCLconvolution_returns_correct_dimensions(self):
        self.out_channels = 8
        self.in_channels = 1
        # (batch_size, channel, height, width)
        input_tensor = torch.ones((1, 10, 10))
        weight_tensor = torch.ones((self.batch_size, 1, 3, 3))
        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)

        result = convolution.performOCLconvolution(input_tensor, weight_tensor)
        res_shape = result.shape

        self.assertEqual(res_shape[0], 1)
        self.assertEqual(res_shape[1], 8)
        self.assertEqual(res_shape[2], 8)

    def _9_performOCLconvolution_returnsCorrectResult(self):
        out_channels = 8
        in_channels = 3
        kernel_size = 3
        tensor_width = random.randrange(kernel_size, 20)
        tensor_height = random.randrange(kernel_size, 20)

        input_tensor = torch.ones((in_channels, tensor_height, tensor_width))
        weight_tensor = torch.ones((out_channels, kernel_size, kernel_size))

        convolution = OCL_Conv2D(in_channels, out_channels, kernel_size, use_ocl=True)

        # act
        result = convolution.performOCLconvolution(input_tensor, weight_tensor)

        self.assertIsNotNone(result)

    def test_10_forwardImage_shouldprovideSameResultAsVanilla(self):
        batch_size = 1
        out_channels = 5
        kernel_size = 5

        _, testLoader = getDataLoaders(batch_size, batch_size)

        dataIter = iter(testLoader)
        images, labels = dataIter.next()

        convolution = OCL_Conv2D(images.size()[1], out_channels, kernel_size, use_ocl=True)
        convolutionOrig = Conv2d(images.size()[1], out_channels, kernel_size)

        convolutionOrig.weight.data = convolution.weight.data

        convWeight_plane = convolution.weight[0][0]
        origWeight_plane = convolutionOrig.weight[0][0]
        

        result = convolution.forward(images)
        resultOrig = convolutionOrig.forward(images)

        self.assertIsNotNone(result, "Result should not be None")
        self.assertIsNotNone(resultOrig, "orig Result should not be None")

        self.assertEqual(result.size()[0], resultOrig.size()[0], 
            "Resulting Batch size should be the same")
        self.assertEqual(result.size()[1], resultOrig.size()[1], 
            "Resulting amount of Channels should be the same")
        self.assertEqual(result.size()[2], result.size()[2], 
            "Resulting height should be the same.")
        self.assertEqual(result.size()[3], result.size()[3], 
            "Resulting width should be the same.")

        print("Testing equality of weights.")
        for j in range(convWeight_plane.size()[0]):
            for i in range(convWeight_plane.size()[1]):
                val_ocl = convWeight_plane[j][i]
                val_orig = origWeight_plane[j][i]
                self.assertEqual(val_ocl, val_orig, 
                    "Original and OCL implementation should be the same, but was ocl={:5.23f}, orig={:5.23f}"
                    .format(val_ocl, val_orig))

        print("Testing equality of result.")
        for batchIdx in range(result.size()[0]):
            for channelIdx in range(result.size()[1]):
                for col in range(result.size()[2]):
                    for row in range(result.size()[3]):
                        val_ocl = result[batchIdx][channelIdx][col][row]
                        val_orig = resultOrig[batchIdx][channelIdx][col][row]
                        self.assertEqual(val_ocl, val_orig, 
                        "Result of orig. and ocl impl. should be the same, but was ocl={:5.23f}, orig={:5.23f}"
                        .format(val_ocl, val_orig))
        
if __name__ == "__main__":
    unittest.main()