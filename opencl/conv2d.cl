__kernel void sum(__global const float *temp1_g, __global const float *temp2_g, __global float *result_g) 
        {
            int gid = get_global_id(0);
            result_g[gid] = temp1_g[gid] + temp2_g[gid];
        }


__kernel void conv2d2(__global const float *_kernel, __global const int *k_dim, __global float *input, __global int *i_dim, __global float *result, __global const int *stride)
{
    //int idx_width = get_global_id(0);
    int idx_height = get_global_id(1) * stride[1];

    int idx_width = get_global_id(0) * stride[0];

    int res_width = get_local_size(0);
    int res_height = get_local_size(1);

    if( idx_height == 0 && idx_width == 0) 
    {
        printf("####################################\n");
        printf("Local width = %d, Local height = %d \n", res_width, res_height);
        printf("Dimension of Kernel: %d, %d \n", k_dim[0], k_dim[1]);
        printf("Stride: %d, %d \n", stride[0], stride[1]);
        printf("####################################\n");
    }

    // calc convolution on position

    float temp_sum = 0.0;

    // position of middle of the filter
    int k_middle_width = k_dim[1] / 2;
    int k_middle_height = k_dim[0] / 2;

    int x_input = idx_height + 1;
    int y_input = idx_width + 1;

    for(int j = 0; j < k_dim[1]; j++) // iterate over height
    {
        for(int k = 0; k < k_dim[0]; k++) // iterate over width -> caching
        {
            temp_sum += 
                _kernel[j * k_dim[0] + k] * 
                input[(x_input - k_middle_width + k) + (y_input - k_middle_height + j) * i_dim[0]];
        }
    }

    result[idx_height * res_width + idx_width] = temp_sum;
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