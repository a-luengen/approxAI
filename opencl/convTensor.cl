void print2DFloatArray(float *_array, int height, int width);

__kernel void convolution(
    __global const float *_kernel, 
    __global const int *k_dim,
    __global const float *_input, 
    __global const int *i_dim,
    __global float *_output,
    __global const int *o_dim,
    __global const int *stride
) {

    bool DEBUG = false;

    int input_channels = k_dim[0];

    int kernel_mid_height = k_dim[1] / 2;
    int kernel_mid_width = k_dim[2] / 2;

    // possition in a single plane of the input, a Worker-Thread is responsible for
    int height_pos = get_global_id(0) * stride[0] + kernel_mid_height;
    int width_pos = get_global_id(1) * stride[1] + kernel_mid_width;

    if(DEBUG && get_global_id(0) == 0 && get_global_id(1) == 0) {
        printf("########Input values##############\n");
        printf("Kernel dimensions: %d, %d, %d\n", k_dim[0], k_dim[1], k_dim[2]);
        printf(" Input dimensions: %d, %d, %d\n", i_dim[0], i_dim[1], i_dim[2]);
        printf("Output dimensions: %d, %d\n", o_dim[0], o_dim[1]);
        printf("           Stride: %d, %d\n", stride[0], stride[1]);
        printf("        Work Size: %zu, %zu, %zu\n", get_global_size(0), get_global_size(1), 0L);
        printf("##################################\n");
    }

    if (height_pos - 1 < o_dim[0] && width_pos - 1 < o_dim[1]) {
        float temp_sum = 0.0;

        
        // iterate through input-channels
        for(int i = 0; i < input_channels; i++) {
            float result = 0.0;

            
            for(int j = 0; j < k_dim[1]; j++) { // height
                
                for(int k = 0; k < k_dim[2]; k++) { // width
                    
                    int kernel_inputchannel_start_index = i * ( k_dim[1] * k_dim[2] );
                    int kernel_height_start_index = kernel_inputchannel_start_index + j * k_dim[1];
                    int kernel_width_pos_index = kernel_height_start_index + k;
                    
                    int input_channel_start_index = i * (i_dim[1] * i_dim[2]);

                    int input_height_start_index = input_channel_start_index + (height_pos + j - kernel_mid_height) * i_dim[1];

                    int input_width_start_index = input_height_start_index + width_pos + k - kernel_mid_width;
                    
                    float kernel_value = _kernel[kernel_width_pos_index];
                    float input_value = _input[input_width_start_index];

                    result += kernel_value * input_value;
                    
                    // if(height_pos == 1 && width_pos == 1) {
                        
                    //     // printf("Read value from kernel at: channel_index=%d height_index=%d width_index=%d\n",
                    //     //     input_channel_start_index,
                    //     //     input_height_start_index,
                    //     //     input_width_start_index
                    //     // );

                    //     // printf("Kernel value=%.5f | Input value=%.5f | result=%.5f\n", 
                    //     //     kernel_value,
                    //     //     input_value,
                    //     //     result
                    //     // );
                    // }               

                    //result += _kernel[j * k_dim[1] + k] 
                    //    * input[
                    //        (height_pos - kernel_mid_height + k) 
                    //        + (width_pos - kernel_mid_width + j) * i_dim[1] 
                    //        + i * i_dim[1] * i_dim[2]];
                }
            }
            temp_sum += result;
        }

        // int output_height_pos = height_pos;
        // int output_width_pos = height_pos + width_pos * o_dim[0];
        // if(height_pos == 13 && width_pos == 13) {
        //     printf("Output height=%d, width=%d, value=%.5f \n",
        //         output_height_pos,
        //         output_width_pos,
        //         temp_sum
        //     );
        // }
        if(DEBUG && height_pos == 1 && width_pos == 1) {
            printf("Writing value=%.5f on height=%d, width=%d, index=%d\n", 
                temp_sum, 
                height_pos - 1, 
                width_pos - 1, 
                (height_pos - 1) * o_dim[0] + (width_pos - 1));
        }
        _output[get_global_id(0) * o_dim[0] + get_global_id(1)] = temp_sum;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if(DEBUG && height_pos == 1 && width_pos == 1) {
        printf("\n[");
        for (int i = 0; i < o_dim[0]; i++) {
            printf("[");
            for(int j = 0; j < o_dim[1]; j++) {
                printf("%.5f, ", _output[i * o_dim[0] + j]);
            }
            printf("],\n");
        }
        printf("]\n");
    }
}