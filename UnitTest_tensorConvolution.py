import unittest
import torch
import random
from torch.nn import Conv2d
import numpy as np

from net.OCL_Convolution import OCL_Convolution

class Test(unittest.TestCase):

    batch_size = 1
    out_channels = 2
    in_channels = 4
    input_width = 12
    input_height = 12
    kernel_height = 3
    kernel_width = 3

    def getTestTensorAndWeight(self):
        input_tensor = torch.ones((self.batch_size, self.in_channels, self.input_height, self.input_width))
        input_weight = torch.ones((self.out_channels, self.in_channels, self.kernel_height, self.kernel_width))
        return (input_tensor, input_weight)

    def getRandomTestTensorAndWeight(self):
        input_tensor = torch.rand((self.batch_size, self.in_channels, self.input_height, self.input_width))
        input_weight = torch.rand((self.out_channels, self.in_channels, self.kernel_height, self.kernel_width))
        return (input_tensor, input_weight)

    def getTorchConvolutionWithWeights(self, weights):
        torch_conv = torch.nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_height, self.kernel_width))
        torch_conv.weight.data = weights
        return torch_conv

    def assertEqualWeights(self, weight1, weight2):
        self.assertEqual(weight1.shape, weight2.shape, "Shapes are different.")
        for i in range(weight1.data.size()[0]):
            for j in range(weight1.data.size()[1]):
                for k in range(weight1.data.size()[2]):
                    for l in range(weight1.data.size()[3]):
                        val_ocl = weight1.data[i][j][k][l]
                        val_orig = weight2.data[i][j][k][l]
                        self.assertEqual(val_ocl, val_orig, 
                            "Original and OCL implementation should be the same, but was ocl={:5.23f}, orig={:5.23f} - iter:{}{}{}{}"
                            .format(val_ocl, val_orig, i, j, k, l))

    def assertEqualTensor(self, tensor1, tensor2):
        self.assertEqual(tensor1.shape, tensor2.shape, "Shapes are different.")
        self.assertEqual(tensor1.dtype, tensor2.dtype, "Datatypes of the tensors are not equal.")
        for i in range(tensor1.shape[0]):
            for j in range(tensor1.shape[1]):
                for k in range(tensor1.shape[2]):
                    for l in range(tensor1.shape[3]):
                        val_ocl = tensor1.detach().numpy()[i][j][k][l]
                        val_orig = tensor2.detach().numpy()[i][j][k][l]
                        self.assertEqual(abs(val_ocl - val_orig) < 0.000001, True, 
                            "Original and OCL implementation should be the same, but was ocl={:5.23f}, orig={:5.23f} - iter:{}-{}-{}-{}"
                            .format(val_ocl, val_orig, i, j, k, l))

    def getOCLandPytorchConvolutionWithSettings(self, in_channels, out_channels, kernel_size, c_bias=False, c_dilation=1, c_padding=0, c_stride=1, c_groups=1):
        ocl_conv = OCL_Convolution(in_channels, out_channels, kernel_size, 
            stride=c_stride, 
            padding=c_padding, 
            dilation=c_dilation, 
            groups=c_groups, 
            bias=c_bias, 
            padding_mode='zeros', 
            use_ocl=True)
        py_conv = Conv2d(in_channels, out_channels, kernel_size, 
            bias=c_bias, 
            dilation=c_dilation, 
            padding=c_padding, 
            stride=c_stride, 
            groups=c_groups)
        return (ocl_conv, py_conv)

    def test_0_createInputAndWeightTensorWithoutException(self):
        input_tensor, input_weight = self.getTestTensorAndWeight()
        self.assertIsNotNone(input_tensor)
        self.assertIsNotNone(input_weight)
        

    def test_1_createTorchConvolutionWithCustomizedWeights_noExcepiontAndCorrectValues(self):
        input_tensor, input_weight = self.getTestTensorAndWeight()
        torch_conv = self.getTorchConvolutionWithWeights(input_weight)

        self.assertIsNotNone(torch_conv)
        self.assertEqual(input_weight.shape, torch_conv.weight.shape, "Weight of artificial and original weight should be the same.")
        for bs in range(input_weight.shape[0]):
            for ic in range(input_weight.shape[1]):
                for ih in range(input_weight.shape[2]):
                    for iw in range(input_weight.shape[3]):
                        self.assertEqual(
                            input_weight[bs][ic][ih][iw], 
                            torch_conv.weight[bs][ic][ih][iw],
                            "Values of the weights should be equal.")

    def test_2_createTensorConvolutionInstance_withoutException(self):
        _, weight_t = self.getTestTensorAndWeight()
        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height, use_ocl=True)
        ocl_conv.weight.data = weight_t

    def test_3_getNumpyOutputDimensions_returnCorrectDimensions_forValidInputWithoutBatchsize(self):
        input_t, weight_t = self.getTestTensorAndWeight()

        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height)

        py_outputDim = Conv2d(self.in_channels, self.out_channels, self.kernel_height).forward(input_t).shape[1:]

        input_shape = input_t.shape[1:] # remove batchsize from input
        weight_shape = weight_t.shape

        output_dimensions = ocl_conv.getNumpyOutputDimensions(input_shape, weight_shape)

        self.assertEqual(type(output_dimensions), type(np.array([])))
        self.assertEqual(output_dimensions[0], py_outputDim[0])
        self.assertEqual(output_dimensions[1], py_outputDim[1])
        self.assertEqual(output_dimensions[2], py_outputDim[2])

    #@unittest.skip("")
    def test_4_forwardOnTensorConvolution_wihtoutException(self):
        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height, use_ocl=True)
        input_t, _ = self.getTestTensorAndWeight()

        ocl_conv.forward(input_t)


    #@unittest.skip("")
    def test_5_forwardOnTensorConvolution_producesSameResultAsPytorch(self):
        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height, use_ocl=True)
        py_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_height, bias=False, dilation=1, padding=0, stride=1, groups=1)
        
        input_t, _ = self.getTestTensorAndWeight()
        py_conv.weight.data = ocl_conv.weight.data

        ocl_result = ocl_conv.forward(input_t)
        py_result = py_conv.forward(input_t)

        self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
        self.assertEqualTensor(ocl_result, py_result)
    
    #@unittest.skip("")
    def test_6_forwardOnTensorConvolution_withRandomInput_producesSameResultAsPytorch(self):
        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height, use_ocl=True)
        py_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_height, bias=False, dilation=1, padding=0, stride=1, groups=1)
        
        input_t, _ = self.getRandomTestTensorAndWeight()
        py_conv.weight.data = ocl_conv.weight.data

        py_result = py_conv.forward(input_t)
        ocl_result = ocl_conv.forward(input_t)


        self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
        self.assertEqualTensor(ocl_result, py_result)

    #@unittest.skip("")
    def test_7_testPadding_withRandomInput_producesSameResultAsPytorch(self):
        test_padding = 5
        ocl_conv = OCL_Convolution(self.in_channels, self.out_channels, self.kernel_height, padding=test_padding, use_ocl=True)
        py_conv = Conv2d(self.in_channels, self.out_channels, self.kernel_height, bias=False, dilation=1, padding=test_padding, stride=1, groups=1)

        input_t, _ = self.getRandomTestTensorAndWeight()
        py_conv.weight.data = ocl_conv.weight.data

        py_result = py_conv.forward(input_t)
        ocl_result = ocl_conv.forward(input_t)
        
        self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
        self.assertEqualTensor(ocl_result, py_result)
    
    #@unittest.skip("")
    def test_8_testStride_withRandomInput_producesSameResultAsPytorch(self):
        
        test_stride = (2, 2)

        ocl_conv, py_conv = self.getOCLandPytorchConvolutionWithSettings(
            self.in_channels,
            self.out_channels,
            self.kernel_width,
            c_stride=test_stride
        )

        input_t, _ = self.getRandomTestTensorAndWeight()
        py_conv.weight.data = ocl_conv.weight.data

        py_result = py_conv.forward(input_t)
        ocl_result = ocl_conv.forward(input_t)

        self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
        self.assertEqualTensor(ocl_result, py_result)

    #@unittest.skip("")
    def test_9_testBias_withRandomInput_producesSameResultAsPytorch(self):
        ocl_conv, py_conv = self.getOCLandPytorchConvolutionWithSettings(
            self.in_channels,
            self.out_channels,
            self.kernel_width,
            c_bias=True
        )

        input_t, _ = self.getRandomTestTensorAndWeight()
        py_conv.weight.data = ocl_conv.weight.data
        py_conv.bias.data = ocl_conv.bias.data
        py_result = py_conv.forward(input_t)
        ocl_result = ocl_conv.forward(input_t)
        self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
        self.assertEqualTensor(ocl_result, py_result)

    #@unittest.skip("")
    def test_10_randomConfig_testingSameResultAsPytorch(self):

        for i in range(5):

            in_channels = random.randint(4, 64)
            out_channels = random.randint(3, 64)
            kernel_width = random.randrange(1, in_channels - 2, 2)
            bias = True if random.randint(1, 2) == 1 else False
            padding = random.randint(0, kernel_width // 2)
            stride = random.randint(1, 4)

            if False:
                print("        Loop: ", i + 1)
                print(" in_channels: ", in_channels)
                print("out_channels: ", out_channels)
                print("kernel_width: ", kernel_width)
                print("        bias: ", bias)
                print("     padding: ", padding)
                print("      stride: ", stride)

            ocl_conv, py_conv = self.getOCLandPytorchConvolutionWithSettings(
                self.in_channels,
                self.out_channels,
                self.kernel_width,
                c_bias=bias,
                c_dilation=1,
                c_padding=0,
                c_stride=stride,
                c_groups=1
            )

            input_t, _ = self.getRandomTestTensorAndWeight()

            py_conv.weight.data = ocl_conv.weight.data
            if bias:
                py_conv.bias.data = ocl_conv.bias.data

            py_result = py_conv.forward(input_t)
            ocl_result = ocl_conv.forward(input_t)
            self.assertEqualWeights(ocl_conv.weight, py_conv.weight)
            self.assertEqualTensor(ocl_result, py_result)

if __name__ == "__main__":
    unittest.main()