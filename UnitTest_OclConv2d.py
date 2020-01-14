import unittest
import torch

# Unit testing for the Convolution implementation

from net.OCL_Conv2D import OCL_Conv2D

class Test(unittest.TestCase):


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
        tensor = torch.ones(10, 10, 1, dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)
        result = convolution.forward(tensor)

        self.assertTrue(isinstance(result, torch.Tensor))
    
    def test_2_convolution_returns_tensor_with_correct_dimensions(self):
        tensor = torch.ones(11, 11, 1, dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, use_ocl=True)
        result = convolution.forward(tensor)
        
        self.assertEqual(result.shape[0], tensor.shape[0] - 2)
        self.assertEqual(result.shape[1], tensor.shape[1] - 2)
        self.assertEqual(result.shape[2], self.out_channels)

    def test_3_convolution_with_stride_returns_correct_dimensions(self):
        tensor = torch.ones(10, 10, 1, dtype=torch.float32)
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = 3
        self.stride = 2

        convolution = OCL_Conv2D(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, use_ocl=True)
        result = convolution.forward(tensor)
        
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result.shape[2], self.out_channels)

if __name__ == "__main__":
    unittest.main()