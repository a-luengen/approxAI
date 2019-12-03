__kernel void sum(__global const float *temp1_g, __global const float *temp2_g, __global float *result_g) 
        {
            int gid = get_global_id(0);
            result_g[gid] = temp1_g[gid] + temp2_g[gid];
        }


__kernel void conv2d(
    __global const float *_kernel, 
    const unsigned int k_width,
    const unsigned int k_height,
    //const int stride,
    //const int dilation,
    __global const float *input, 
    const unsigned int in_height,
    const unsigned int in_width,
    __global float *output,
    const unsigned int out_height,
    const unsigned int out_width
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