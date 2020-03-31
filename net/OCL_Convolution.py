import torch
import torch.nn as nn
import os
import numpy as np

import pyopencl as cl

class OCL_Convolution(nn.Conv2d):
    
    """
    Convolution implemented with OpenCL


    in_channels - Number of channels in the input image
    out_channels - Number of channels produced by the convolution

    use_ocl - Boolean, if set to True, will use OpenCL Convolution instead of PyTorch implementaition


    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', use_ocl=False):
        self.use_ocl = use_ocl
        super(OCL_Convolution, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.context = self.getOclContext()
        self.programm = self.getOCLprogramm(self.context)

    def getOCLprogramm(self, oclContext):
        
        PATH_TO_KERNEL = 'opencl/convTensor.cl'

        # prevent using cached source code, as this may cause 
        # "compiler caching failed exception"
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1' 
        os.environ['PYOPENCL_CTX'] = '0'

        # read kernel file
        f = open(PATH_TO_KERNEL, 'r', encoding='utf-8')
        kernels = ' '.join(f.readlines())
        f.close()

        return cl.Program(oclContext, kernels).build()

    def getOclContext(self):
        platforms = cl.get_platforms()
        return cl.Context(
            dev_type=cl.device_type.ALL,
            properties=[(cl.context_properties.PLATFORM, platforms[0])]
        )

    def getNumpyOutputDimensions(self, input_dim, kernel_dim):
        """
            Given two arrays representing the dimension-size (shape)
            for an input and kernel, calculates the dimension-size (shape)
            when performing a convolution.
            Input: (channel, height, width)
            kernel: (in_channel, height, width)
        """
        assert(len(input_dim) == 3)
        assert(len(kernel_dim) == 3)
        output_height = (input_dim[1] - kernel_dim[1] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (input_dim[2] - kernel_dim[2] + 2 * self.padding[1]) // self.stride[1] + 1

        return np.array([self.out_channels, output_height, output_width], dtype=np.int32)

    def performOCLconvolution(self, input, weight):
        """
            Takes input and weight as torch.tensor type 
            and performs the correct convolution on them
            to produce a torch.tensor type result
            Input tensor has to be (channels, height, width)
            Weights are (out_channels, in_channels , height, width)
            WARNING: do not use 4D tensor with batchsize
        """
        assert(len(input.shape) == 3)
        assert(len(weight.shape) == 4)

        #print("Input Channels:  ", self.in_channels)
        #print("Output Channels: ", self.out_channels)
        #print("Weight shape:    ", weight.shape)
        #print("Input shape:     ", input.shape)
        
        result_tensor = []

        for out_channel in weight:
            #print("Kernel value for out_channel iteration: ", out_channel.shape)
            out_channel_result_tensor = self.OCLconvolution(
                input.detach().numpy(), 
                out_channel.detach().numpy())
            result_tensor.append(torch.tensor(out_channel_result_tensor))

        return torch.stack(result_tensor)

    def OCLconvolution(self, input_3d, kernel_3d):
        """
            Takes 3 dimensional input_3d and 3 dimensional
            weight as numpy array to perform the convolution with
            Returns 3 dimensional numpy-array as result
        """
        assert(len(input_3d.shape) == 3)
        assert(len(kernel_3d.shape) == 3)

        # context and programm
        #ctx = self.getOclContext()
        #prg = self.getOCLprogramm(ctx)
        ctx = self.context
        prg = self.programm
        
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # conversion of data types and creation of buffers

        # 1. Input data, with dimensions
        np_x = input_3d
        if self.padding != 0:
            np.zeros((np_x.shape[0] + self.padding[0], np_x.shape[1] + self.padding[1]), dtype=np.int32)

        np_dim_x = np.array(np_x.shape, dtype=np.int32)

        buffer_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x)
        buffer_dim_x = cl.Buffer(ctx ,mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=np_dim_x)

        # 2. kernel, with dimensions
        np_kernel = kernel_3d
        np_dim_kernel = np.array(np_kernel.shape, dtype=np.int32)

        buffer_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_kernel)
        buffer_dim_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_kernel)

        # 3. Output buffer
        np_dim_output = self.getNumpyOutputDimensions(input_3d.shape, kernel_3d.shape)
        #print('Output Dimensions: ', np_dim_output[1:])
        np_output = np.zeros(np_dim_output[1:], dtype=np.float32)
        
        buffer_output = cl.Buffer(ctx, mf.READ_WRITE, np_output.nbytes)
        buffer_dim_output = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_output[1:])
        # 4. Stride buffer
        buffer_stride = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array((self.stride, self.stride), dtype=np.int32))

        # OpenCL kernel, executed for single result
        convolutionFunct = prg.convolution
        convolutionFunct.set_args(
            buffer_kernel, 
            buffer_dim_kernel, 
            buffer_x,
            buffer_dim_x,
            buffer_output,
            buffer_dim_output,
            buffer_stride)
        #print("Enqueue Kernel with nd-range shape of output: ", np_output.shape)
        cl.enqueue_nd_range_kernel(queue, convolutionFunct, np_output.shape, None)
        cl.enqueue_copy(queue, np_output, buffer_output)
        print("OCL-First value=", np_output[1][2])
        return np_output


    def ocl_conv2d_forward(self, input):
        result = []
        for batch in input:
            tempRes = self.performOCLconvolution(batch, self.weight)
            result.append(tempRes)

        return torch.stack(result)

    def forward(self, x):
        if self.use_ocl == True:
            return self.ocl_conv2d_forward(x)
        else:
            return super(OCL_Convolution, self).forward(x)
        
