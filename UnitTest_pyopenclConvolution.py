import unittest

import os
import numpy as np
import pyopencl as cl
import torch

class Test(unittest.TestCase):

    PATH_TO_KERNEL = 'opencl/conv2d.cl'
    ARRAY_SIZE = 10

    def compileAndGetOclKernelProgramm(self, context):
        """

        """
        # prevent using cached source code, as this may cause 
        # "compiler caching failed exception"
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1' 
        os.environ['PYOPENCL_CTX'] = '0'

        # read kernel file
        f = open(self.PATH_TO_KERNEL, 'r', encoding='utf-8')
        kernels = ' '.join(f.readlines())
        f.close()

        return cl.Program(context, kernels).build()
    
    def getOclContext(self):
        platforms = cl.get_platforms()
        return cl.Context(
            dev_type=cl.device_type.ALL, 
            properties=[(cl.context_properties.PLATFORM, platforms[0])])


    def test_0_compile_without_error(self):
        """
            Simple test to asure, that the opencl kernel is building without errors.
        """
        self.compileAndGetOclKernelProgramm(self.getOclContext())

    def test_1_executes_hello_world_kernel(self):
        """
            Test that a simple hello-world kernel is executed correctly.
            Hello-World example calculates element wise sum of two
            one-dimensional numpy arrays with type float32.
        """
        context = self.getOclContext()
        programm = self.compileAndGetOclKernelProgramm(context)
        queue = cl.CommandQueue(context)
        mf = cl.mem_flags

        arrayA = np.ones(self.ARRAY_SIZE, dtype=np.float32)
        arrayB = np.ones(self.ARRAY_SIZE, dtype=np.float32)
        arrayC = np.empty_like(arrayA)
        
        bufferA = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arrayA)
        bufferB = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arrayB)
        bufferC = cl.Buffer(context, mf.WRITE_ONLY, arrayC.nbytes)

        programm.sum(queue, arrayA.shape, None, bufferA, bufferB, bufferC)
        cl.enqueue_copy(queue, arrayC, bufferC)

        for a, b, c in zip(arrayA, arrayB, arrayC):
            self.assertEqual(a + b, c)
    
    def test_2_executes_convolution_kernel_with_3x3_kernel(self):
        """
            Executes a basic 2-dimensional convolution with a 
            3x3 kernel on a nxn Input Matrix containing only ones.
        """
        self.ARRAY_SIZE = 4
        context = self.getOclContext()
        programm = self.compileAndGetOclKernelProgramm(context)
        queue = cl.CommandQueue(context)
        mf = cl.mem_flags

        input = np.ones((self.ARRAY_SIZE, self.ARRAY_SIZE), dtype=np.float32)
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1,1] = 0.5
        output = np.zeros((self.ARRAY_SIZE - 2, self.ARRAY_SIZE - 2), dtype=np.float32)


        bufferInput = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input)
        bufferKernel = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)
        bufferOutput = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

        bufferInputDim = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(input.shape, dtype=np.int32))
        bufferKernelDim = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(kernel.shape, dtype=np.int32))
        bufferStride = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array((1,1), dtype=np.int32))

        convolutionFunct = programm.conv2d2
        convolutionFunct.set_args(
            bufferKernel, 
            bufferKernelDim, 
            bufferInput, 
            bufferInputDim, 
            bufferOutput, 
            bufferStride)
        cl.enqueue_nd_range_kernel(queue, convolutionFunct, output.shape, None)
        cl.enqueue_copy(queue, output, bufferOutput)

        for a in output:
            for b in a: 
                self.assertEqual(b - 0.5, 0.0)


if __name__ == '__main__':
    unittest.main()
