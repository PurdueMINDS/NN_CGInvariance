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
    """
    CG-invariant sequence layer.
    Input is a sequence of length n and dimension d (can be intermediate representations).
    The layer takes as parameter a list of m groups (respective Reynolds operator functions) applied on sequences.
    For example, we will typically use \binom{n}{2} transposition groups.
    After loading the basis for all subspaces (using invariantSubspaces module), the final weight vector is computed
    as the linear combination of the basis (with learnable coefficients). Output of the layer is the dot product of the
    final weight vector and the input (plus a bias).
    """
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

        self.input_size = input_size    # Sequence length
        self.input_dimension = input_dimension  # Dimension of each element

        # A list of m groups for this layer given in the form of their respective Reynolds operator functions
        # For example, all transposition groups: [IS.G_permutation_ij]. See usage below.
        self.invariant_transforms = invariant_transforms

        # Different set of weights for each input dimension (True) or shared (False).
        self.weightsAcrossDims = weightsAcrossDims
        self.hidden_dimension = hidden_dimension    # Output dimension of the layer

        # Get the basis of subspaces for given list of groups.
        self.invariant_transforms, basisList, self.basisConfigs = self.getBasis(precomputed_basis_folder)
        basisList = [torch.Tensor(basis.T) for basis in basisList]
        self.basisShapes = [basis.shape[0] for basis in basisList]
        self.basis = torch.cat(basisList)

        # Weights are the parameters of the linear combination corresponding to each basis vector.
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
        # Initialize the weights and biases.
        if self.weightsAcrossDims:
            stdv = 1.0 / math.sqrt(self.input_size * self.input_dimension)
        else:
            stdv = 1.0 / math.sqrt(self.input_size)

        self.weights.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_()

    def getBasis(self, precomputed_basis_folder="."):
        # Using Wshape = (self.input_size, 1) instead of (self.input_size, d).
        # Reason: We can use basis vectors across all input dimension but the
        #         learnable weights can be different across the dimensions).
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

        #   Obtain the final weight as a linear combination of learnable weights and the bases.
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

        # Dot product of the final weight and the input.
        # Output shape: torch.Size([minibatch, input_size=basisDim, inputDimension])
        out = torch.mul(fullSub, input)
        out = out.sum(dim=[-2, -1])

        # One bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def penalty(self, mode=None, T=None, debug=False):
        # Penalty depends on the invariances of subspaces that were used.
        # Penalty is the least if only the fully invariant subspace is used.
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

