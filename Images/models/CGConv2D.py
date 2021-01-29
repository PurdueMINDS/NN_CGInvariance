import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
import invariantSubspaces as IS
import invarianceUtils as IUtils
from Common import utils
import os

loadedBases = {}

class CGConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        invariant_transforms,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
        precomputed_basis_folder=".",
        penaltyAlpha=0,
    ):
        super().__init__()

        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.invariant_transforms = invariant_transforms
        self.penaltyAlpha = penaltyAlpha

        if IS.G_color_permutation in self.invariant_transforms:
            # Channels is part of the basis
            self.Wshape = (self.in_channels, *self.kernel_size)
            sameBasisAcross = 1
        else:
            # Same basis is enough for all channels
            self.Wshape = self.kernel_size
            sameBasisAcross = self.in_channels

        basisFileName = IS._getBasisFileName(listT=self.invariant_transforms, Wshape=self.Wshape)
        if basisFileName not in loadedBases:
            self.invariant_transforms, basisList, self.basisConfigs = self.getBasis(precomputed_basis_folder)
            basisList = [torch.Tensor(basis.T) for basis in basisList]
            self.basisShapes = [basis.shape[0] for basis in basisList]
            self.basis = torch.cat(basisList).to(utils.getDevice())
            loadedBases[basisFileName] = (self.invariant_transforms, self.basisConfigs, self.basisShapes, self.basis)

        self.invariant_transforms, self.basisConfigs, self.basisShapes, self.basis = loadedBases[basisFileName]

        # The dimension one at the end is needed in the tensor, as we will use it to broadcast the
        #   coefficient parameters over the basis dimension
        self.weights = nn.Parameter(torch.Tensor(out_channels, sameBasisAcross, self.basis.shape[0], 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)

        self.weights.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def getBasis(self, precomputed_basis_folder="."):
        listT, basis, basisConfigs = IS.getAllBasis(
            folder=precomputed_basis_folder,
            listT=self.invariant_transforms,
            Wshape=self.Wshape,
        )

        return listT, basis, basisConfigs


    def forward(self, input):
        if self.basis.device != input.device:
            self.basis = self.basis.to(input.device)

        fullSub = torch.mul(self.weights, self.basis)
        fullSub = fullSub.sum(dim=-2)

        # Unvectorize
        #   Result after unvectorize: torch.Size([out_channels, in_channels, kernel_size[0], kernel_size[1]])
        fullSub = fullSub.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        # Input shape: torch.Size([minibatch, in_channels, iH, iW])

        # Output shape: torch.Size([minibatch, out_channels, oH, oW])
        out = F.conv2d(
            input, weight=fullSub, bias=None, stride=self.stride, padding=self.padding
        )

        # One bias per channel
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            out = out + bias

        return out

    def penalty(self, mode=None, T=None):
        return IUtils.invariancePenalty(self, mode=mode, T=T)

if __name__ == '__main__':
    # Usage
    basisDir = 'data/basis'
    os.makedirs(basisDir, exist_ok=True)
    layer = CGConv2D(3, 10, kernel_size=3, invariant_transforms=[IS.G_rotation, IS.G_verticalFlip, IS.G_color_permutation], precomputed_basis_folder=basisDir)
    layer = layer.to("cuda")
    X = torch.rand((100, 3, 28, 28)).to("cuda")
    outs = layer.forward(X)
    print(outs.shape)

