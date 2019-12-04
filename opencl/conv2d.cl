__kernel void sum(__global const float *temp1_g, __global const float *temp2_g, __global float *result_g) 
        {
            int gid = get_global_id(0);
            result_g[gid] = temp1_g[gid] + temp2_g[gid];
        }


__kernel void conv2d2(__global float *_kernel, __global int *k_dim, __global float *input, __global int *i_dim, __global float *result)
{
    int idx_width = get_global_id(0);
    int idx_height = get_global_id(1);
    int w_dim = get_work_dim();
    int get_global_size();
    result[]
    /*
    int out_width = k_dim[0] - 2;
    int out_height = k_dim[1] - 2;

    for(int i = 0; i < out_height; i++) 
    {
        for(int j = 0; j < out_width; i++) 
        {
            result[i + j * out_width] = 0.0;
        }
    }
    result[gid] = 0.0;
    */

}

__kernel void conv2d(
    __global const float *_kernel, 
    int k_width,
    unsigned int k_height,
    //const int stride,
    //const int dilation,
    __global const float *input, 
    unsigned int in_height,
    unsigned int in_width,
    __global float *output,
    unsigned int out_height,
    unsigned int out_width
) {

    for(unsigned int y = 0; y < (in_width - k_height - 1); y++) 
    {
        for(unsigned int x = 0; x < (in_height - k_width - 1); x++) 
        {
            float sum = 0.0;

            for(unsigned fY = y; fY < y + k_height; fY++) 
            {
                for(unsigned fX = x; fX < x + k_width; fX++) 
                {
                    sum += input[(y + fY) * in_width + x + fX] * _kernel[fY * k_height + fX];
                }
            }
            sum /= k_height * k_width;
            output[ y * out_width + x] = sum;
        }
    }

}