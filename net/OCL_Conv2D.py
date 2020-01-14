import torch
import torch.nn as nn
import os
import numpy as np

import pyopencl as cl

class OCL_Conv2D(nn.Conv2d):
    
    """
    Convolution implemented with OpenCL


    in_channels - Number of channels in the input image
    out_channels - Number of channels produced by the convolution

    use_ocl - Boolean, if set to True, will use OpenCL Convolution instead of PyTorch implementaition


    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_ocl=False):
        self.use_ocl = use_ocl
        super(OCL_Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def getOCLprogramm(self, oclContext):
        
        PATH_TO_KERNEL = 'opencl/conv2d.cl'

        print("Performing OCL Convolution with impl: " + PATH_TO_KERNEL)

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

    def getNumpyOutputDimensions(self, input_dim, kernel_dim):
        output_width = (input_dim[0] - kernel_dim[0] + 2 * self.padding) // self.stride + 1
        output_height = (input_dim[1] - kernel_dim[1] + 2 * self.padding) // self.stride + 1
        return np.array([output_width, output_height], dtype=np.int32)

    def getOutputDimensions(self, input, weight):
        output_width = (input.shape[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_height = (input.shape[1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        return np.array([output_width, output_height], dtype=np.int32)

    def performOCLconvolution(self, input, weight, output_dim):

        for kernel_plane in weight:
            output_plane = np.zeros(output_dim, dtype=np.foat32)

            for input_plane in input:
                temp_res = self.OCLconvolution2D(input_plane, kernel_plane) 
                # add temp_res onto output_plane
                output_plane = output_plane.__add__(temp_res)
            # insert output_plane into correct channel

        return self.OCLconvolution2D(self, x, self.kernel)

    def OCLconvolution2D(self, input_2d, kernel_2d):

        # context and programm
        ctx = cl.create_some_context()
        prg = self.getOCLprogramm(ctx)
        
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # conversion of data types and creation of buffers

        # 1. Input data, with dimensions
        np_x = input.numpy()
        if self.padding != 0:
            np.zeros((np_x.shape[0] + self.padding, np_x.shape[1] + self.padding), dtype=np.int32)

        np_dim_x = np.array(np_x.shape, dtype=np.int32)

        buffer_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x)
        buffer_dim_x = cl.Buffer(ctx ,mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=np_dim_x)

        # 2. kernel, with dimensions
        np_kernel = kernel.numpy()
        np_dim_kernel = np.array(np_kernel.shape, dtype=np.int32)

        buffer_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_kernel)
        buffer_dim_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_kernel)

        # 3. Result buffer
        np_dim_output = self.getNumpyOutputDimensions(np_dim_x, np_dim_kernel)
        np_output = np.array(np_dim_output.shape, dtype=np.int32)

        buffer_output = cl.Buffer(ctx, mf.READ_WRITE, None)

        # buffers
        

        #buffer_kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.weight.detach().numpy())

        #buffer_result = cl.Buffer(ctx, mf.WRITE_ONLY, x.numpy().nbytes)

        # OpenCL kernel, executed for single result
        prg.conv2d2()

        return result


    def ocl_conv2d_forward(self, input, weight):
        output_dim = self.getOutputDimensions(input, weight)
        return torch.ones((output_dim[0], output_dim[1], self.out_channels), dtype=torch.float32)

    def forward(self, x):
        if self.use_ocl == True:
            return self.ocl_conv2d_forward(x, self.weight)
        else:
            return super(OCL_Conv2D, self).forward(x)
        
