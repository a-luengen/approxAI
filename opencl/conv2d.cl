__kernel void sum(__global const float *temp1_g, __global const float *temp2_g, __global float *result_g) 
        {
            int gid = get_global_id(0);
            result_g[gid] = temp1_g[gid] + temp2_g[gid];
        }