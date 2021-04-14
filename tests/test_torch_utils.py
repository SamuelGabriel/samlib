import unittest

import torch
from torch import nn

from samlib import torch_utils
from . import utils as U


class TorchUtils(unittest.TestCase):
    def test_get_gradients(self):
        model = nn.Sequential(nn.Linear(10,10,bias=False),nn.Linear(10,10,bias=False),nn.Linear(10,10,bias=False))
        grads = []
        for i,m in enumerate(model):
            w = m.weight

            if i == 0:
                g = None
            else:
                g = torch.rand_like(w)
            grads.append(g)

            w.grad = g


        gotten_grads = torch_utils.get_gradients(model, copy=False, as_vector=False)
        self.assertEqual(len(grads),len(gotten_grads))
        for g, g_ in zip(grads, gotten_grads): self.assertIs(g, g_)

        gotten_grads = torch_utils.get_gradients(model, copy=True, as_vector=False)
        self.assertEqual(len(grads),len(gotten_grads))
        for g, g_ in zip(grads[1:], gotten_grads[1:]):
            self.assertTrue((g == g_).all())
            self.assertFalse(g is g_)

        grad_vec = torch_utils.get_gradients(model, copy=True, as_vector=True)
        self.assertTrue((grad_vec == torch.cat([grads[1].flatten(),grads[2].flatten()])).all())

    def test_de_normalize(self):
        img_tensor = torch.rand(4,100,100)
        mean = torch.rand(4)
        std = torch.rand(4)

        self.assertTrue(U.almost_equal(img_tensor, torch_utils.denormalize(torch_utils.normalize(img_tensor, mean, std), mean, std)))
        self.assertTrue(U.almost_equal(img_tensor, torch_utils.normalize(torch_utils.denormalize(img_tensor, mean, std), mean, std)))


if __name__ == '__main__':
    unittest.main()
