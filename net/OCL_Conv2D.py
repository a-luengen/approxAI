import torch
import torch.nn as nn
import numpy

class OCL_Conv2D(nn.Conv2d):
    
    """
    Convolution implemented with OpenCL
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_ocl=False):
        
        self.use_ocl = use_ocl
        
        super(OCL_Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):

        if self.use_ocl:
            np_x = x.detach().numpy()
            
            print(x.shape)
            print(x.dtype)
            print(np_x.shape)
            print(np_x.dtype)
            return super(OCL_Conv2D, self).forward(torch.from_numpy(np_x))
        else:
            print(x.shape)
            print(x.dtype)
            return super(OCL_Conv2D, self).forward(x)


    def printAttr(self):
        print("Attributes for this convolution")
        print(f'in_ch:%d,  out_ch:%d, knl_s:%s' % (self.in_channels, self.out_channels, self.kernel_size))
        print(f'strd: %s, pad: %s, dila: %s' % (self.stride, self.padding, self.dilation))
        print(f'grps: %d, pad_md: %s, ocl: %s' % (self.groups, self.padding_mode, self.use_ocl))
