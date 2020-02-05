import unittest
import torch
import numpy as np


class Test(unittest.TestCase):

    def test_0_create_new_tensor_with_new_dimension(self):
        tensor1 = torch.ones((10, 10))
        tensor2 = torch.zeros((10, 10))

        tensor3 = torch.stack([tensor1, tensor2])

        self.assertEqual(len(tensor3.shape), 3)
        self.assertEqual(tensor3.shape[0], 2)
        self.assertEqual(tensor3.shape[1], 10)
        self.assertEqual(tensor3.shape[2], 10)

    def test_1_stack_three_tensors(self):
        channel_size = 3
        height = 10
        width = 10
        
        
        tensor1 = torch.ones((height, width))
        tensor2 = torch.zeros((height, width))
        tensor3 = torch.ones((height, width))

        temp = [tensor1]
        temp.append(tensor2)
        temp.append(tensor3)

        tensor4 = torch.stack(temp)

        self.assertEqual(len(tensor4.shape), channel_size)

    def test_2_stack_tensors_made_from_numpy_arrays(self):
        height = 10
        width = 10
        channel_size = 2

        np_t1 = np.ones((height, width))
        np_t2 = np.zeros((height, width))

        temp = [torch.tensor(np_t1)]
        temp.append(torch.tensor(np_t2))

        tensor4 = torch.stack(temp)
        
        self.assertEqual(len(tensor4.shape), 3)
        self.assertEqual(tensor4.shape[0], channel_size)
        self.assertEqual(tensor4.shape[1], height)
        self.assertEqual(tensor4.shape[2], width)

if __name__ == "__main__":
    unittest.main()