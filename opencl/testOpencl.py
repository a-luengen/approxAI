import numpy as np 
import pyopencl as cl
import os


if __name__ == "__main__":
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    # prevent using cached source code, as this may cause compiler caching failed exception
    os.environ['PYOPENCL_NO_CACHE'] = '1' 
    os.environ['PYOPENCL_CTX'] = '0'

    print("Loading kernel...")
    f = open('testOpencl.cl', 'r', encoding='utf-8')
    kernels = ' '.join(f.readlines())
    f.close()

    temp1_np = np.random.rand(50000000).astype(np.float32)
    temp2_np = np.random.rand(50000000).astype(np.float32)

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

    print(result_np - (temp1_np + temp2_np))
    print(np.linalg.norm(result_np - (temp1_np + temp2_np)))
    assert np.allclose(result_np, temp1_np + temp2_np)