import torch
import torch.nn as nn
import torch.nn.functional as F


class SR(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.register_buffer("h", torch.ones(1, 1, s, s) / (s ** 2))

    def _get_kernel(self, x):
        C = x.shape[1]
        return self.h.repeat(C, 1, 1, 1)

    def forward(self, x):  # y = Ax
        h = self._get_kernel(x)
        return F.conv2d(x, h, stride=self.s, groups=x.shape[1])

    def adjoint(self, y):  # x = A^T y
        h = self._get_kernel(y)
        return F.conv_transpose2d(y, h, stride=self.s, groups=y.shape[1])

    def backprojection(self, y):
        """Return a normalized image-space backprojection."""
        return self.adjoint(y) * (self.s ** 2)

    def transpose(self, y):
        return self.adjoint(y)

    def pseudoinverse(self, y):
        return self.backprojection(y)
