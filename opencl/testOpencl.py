import numpy as np 
import pyopencl as cl
import os
import torch

PATH_TO_KERNEL = 'conv2d.cl'

def testSum():
    print("Loading kernel...")
    #f = open('testOpencl.cl', 'r', encoding='utf-8')
    f = open(PATH_TO_KERNEL, 'r', encoding='utf-8')
    kernels = ' '.join(f.readlines())
    f.close()

    SIZE = 5

    temp1_np = np.random.rand(SIZE).astype(np.float32)
    temp2_np = np.random.rand(SIZE).astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    temp1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=temp1_np)
    temp2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=temp2_np)

    programm = cl.Program(ctx, kernels).build()
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, temp1_np.nbytes)

    programm.sum(queue, temp1_np.shape, None, temp1_g, temp2_g, result_g)

    result_np = np.empty_like(temp1_np)
    cl.enqueue_copy(queue, result_np, result_g)

    print(temp1_np)
    print(temp2_np)
    print(result_np)
    print(result_np - (temp1_np + temp2_np))
    print(np.linalg.norm(result_np - (temp1_np + temp2_np)))
    assert np.allclose(result_np, temp1_np + temp2_np)

def testConvolution():
    # read kernel file
    f = open(PATH_TO_KERNEL, 'r', encoding='utf-8')
    kernels = ' '.join(f.readlines())
    f.close()

    # create context, queue, buffers and compile kernels
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, kernels).build()

    # init parameters for kernel-call

    cX = torch.ones(10, 10)
    print("Input x shape - " + str(cX.shape))
    print(cX)
    cKernel = getKernel(4)
    print("Input kernel shape - " + str(cKernel.shape))
    print(cKernel)
    cOutput = torch.zeros(8, 8)
    print("Output shape - " + str(cOutput.shape))

    # convert Tensors into usabel np_arrays
    np_cX = cX.numpy()
    np_cKernel = cKernel.numpy()
    np_cOutput = cOutput.numpy()
    print("Numpy cX - ") 
    print(str(np_cX))
    print("Numpy cX dType - " + str(np_cX.dtype))
    print("Numpy cKernel - ")
    print(str(np_cKernel))
    print("Numpy cKernel dType - " + str(np_cKernel.dtype))

    np_dim_cX = np.array(np_cX.shape, dtype=np.int32) # fits device integer bit-length
    print("Dimensions of np_cX - " + str(np_dim_cX))
    print("Dtype of np_cX - " + str(type(np_dim_cX[0])))
    np_dim_cKernel = np.array(np_cKernel.shape, dtype=np.int32) # fits device integer bit-length
    print("Dimensions of np_cKernel - " + str(np_dim_cKernel))
    print(np_cKernel.shape)
    print(type(np_dim_cKernel[0]))

    # copy np_arrays into buffers of device

    # input
    buf_cX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_cX)
    # input dimension
    buf_dim_cX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_cX)
    # kernel
    buf_cKernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_cKernel)
    # kernel dimension
    buf_dim_cKernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dim_cKernel)
    # output
    buf_cOutput = cl.Buffer(ctx, mf.WRITE_ONLY, np_cOutput.nbytes)

    ######
    # Options
    ######
    # stride
    np_stride = np.array([1, 1], dtype=np.int32)

    buf_stride = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_stride)

    print("Calling Kernel with global_work_size: " + str(np_cOutput.shape[0] * np_cOutput.shape[1]))


    convKernel = prg.conv2d2
    convKernel.set_args(buf_cKernel, buf_dim_cKernel, buf_cX, buf_dim_cX, buf_cOutput, buf_stride)
    ev = cl.enqueue_nd_range_kernel(queue, convKernel, np_cOutput.shape, None)
    
    #prg.conv2d2(queue, np_cOutput.shape, None,
    #    buf_cX, buf_dim_cX,
    #    buf_cKernel, buf_dim_cKernel,
    #    buf_cOutput)

    cl.enqueue_copy(queue, np_cOutput, buf_cOutput)
    print(type(np_cOutput))
    print(np_cOutput.dtype)
    print(np_cOutput)


def getKernel(num):
    if num == 0:
        return torch.ones(3, 3)
    elif num == 1:
        return torch.zeros(3,3)
    elif num == 2:
        return torch.ones(5,5)
    else:     
        return torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

if __name__ == "__main__":
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    # prevent using cached source code, as this may cause compiler caching failed exception
    os.environ['PYOPENCL_NO_CACHE'] = '1' 
    os.environ['PYOPENCL_CTX'] = '0'

    testSum()
    testConvolution()

