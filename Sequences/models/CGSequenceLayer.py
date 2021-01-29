import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import invariantSubspaces as IS
import invarianceUtils as IUtils
import itertools
from functools import partial
import os


class CGSequenceLayer(nn.Module):
    def __init__(
            self,
            input_size,
            input_dimension,
            hidden_dimension,
            invariant_transforms,
            precomputed_basis_folder=".",
            weightsAcrossDims=False,
            bias=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_dimension = input_dimension
        self.invariant_transforms = invariant_transforms
        self.weightsAcrossDims = weightsAcrossDims
        self.hidden_dimension = hidden_dimension

        self.invariant_transforms, basisList, self.basisConfigs = self.getBasis(precomputed_basis_folder)
        basisList = [torch.Tensor(basis.T) for basis in basisList]
        self.basisShapes = [basis.shape[0] for basis in basisList]
        self.basis = torch.cat(basisList)

        # The dimension one at the end is needed in the tensor, as we will use it to broadcast the
        #   coefficient parameters over the basis dimension
        if weightsAcrossDims:
            self.weights = nn.Parameter(torch.Tensor(self.hidden_dimension, self.input_dimension, self.basis.shape[0], 1))
        else:
            self.weights = nn.Parameter(torch.Tensor(self.hidden_dimension, 1, self.basis.shape[0], 1))

        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weightsAcrossDims:
            stdv = 1.0 / math.sqrt(self.input_size * self.input_dimension)
        else:
            stdv = 1.0 / math.sqrt(self.input_size)

        self.weights.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def getBasis(self, precomputed_basis_folder="."):
        # Using Wshape = (self.input_size, 1) instead of (self.input_size, d).
        # Same basis across all dimension (with potentially different weights).
        listT, basis, basisConfigs = IS.getAllBasis(
            folder=precomputed_basis_folder,
            listT=self.invariant_transforms,
            Wshape=(self.input_size, 1),
            powerSetMethod=IS.indexPowerSetTranspositions,
        )

        return listT, basis, basisConfigs

    def forward(self, input):
        if self.basis.device != input.device:
            self.basis = self.basis.to(input.device)

        #   Weight Coefficients : torch.Size([hiddenDimension, inputDimension, basisDim, 1])
        #   Basis : torch.Size([basisDim, basisDim])
        #   Result after mul: torch.Size([hiddenDimension, inputDimension, basisDim, basisDim])
        #   Result after sum: torch.Size([hiddenDimension, inputDimension, basisDim])
        fullSub = torch.mul(self.weights, self.basis)
        fullSub = fullSub.sum(dim=-2)

        #   Result after transpose: torch.Size([hiddenDimension, basisDim, inputDimension])
        fullSub = fullSub.transpose(-2, -1)

        # Input shape: torch.Size([minibatch, input_size=basisDim, inputDimension])
        # After unsqueeze: torch.Size([minibatch, 1, input_size=basisDim, inputDimension])
        input = input.unsqueeze(1)

        # Output shape: torch.Size([minibatch, input_size=basisDim, inputDimension])
        out = torch.mul(fullSub, input)
        out = out.sum(dim=[-2, -1])

        # One bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def penalty(self, mode=None, T=None, debug=False):
        return IUtils.invariancePenalty(self, mode=mode, T=T, debug=debug)

if __name__ == '__main__':
    # Usage
    basisDir = 'data/basis'
    os.makedirs(basisDir, exist_ok=True)

    n = 5
    d = 128

    # Transposition groups
    listT = []
    for ij in list(itertools.combinations(range(n), 2)):
        f = partial(IS.G_permutation, pos=ij)
        f.__name__ = f"({ij[0]} {ij[1]})"
        listT.append(f)

    layer = CGSequenceLayer(input_size=n, input_dimension=d, hidden_dimension=128, invariant_transforms=listT, precomputed_basis_folder=basisDir)
    layer = layer.to("cuda")
    X = torch.rand((10, n, d)).to("cuda")
    outs = layer.forward(X)
    print(outs.shape)

