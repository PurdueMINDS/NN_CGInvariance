import numpy as np
import torch


def invariancePenalty(layer, mode=None, T=None, debug=False):
    # Input: Basis configurations and list of weights corresponding to each basis.
    def _l0_proxy(w, mode, T):
        if mode is None or mode.lower() == "true":
            return w.norm(p=0)
        elif mode.lower() == "simple":
            return T * w / (T * w + 1)
        elif mode.lower() == "sigmoid":
            return 2 * torch.sigmoid(T * w) - 1
        else:
            raise NotImplementedError

    invarianceOfSubspaces = [len(config.split('_')) if config!='' else 0 for config in layer.basisConfigs]

    i = -1
    prevBasisShape = 0
    nCumulativeSubspaces = []       # Cumulative number of subspaces till level i.
    sigma = []                      # Tracks whether i-th invariance level is used or not.
    nSubspacesUsed = []             # Tracks the number of subspaces used in each level.
    nSubspacesUsedSoft = []
    while i < len(invarianceOfSubspaces) - 1:
        nCumulativeSubspaces.append(0)
        nSubspacesUsed.append(0)
        nSubspacesUsedSoft.append(0)
        current = invarianceOfSubspaces[i+1]
        normSum = 0
        while i < len(invarianceOfSubspaces)-1 and invarianceOfSubspaces[i+1] == current:
            index = torch.arange(prevBasisShape, prevBasisShape+layer.basisShapes[i+1], device=layer.weights.device)
            normSumi = layer.weights.index_select(dim=-2, index=index).norm(p=2)
            normSum += normSumi
            nCumulativeSubspaces[-1] += 1
            if normSumi > 0:
                nSubspacesUsed[-1] += 1
            nSubspacesUsedSoft[-1] += _l0_proxy(normSumi, mode=mode, T=T)

            prevBasisShape += layer.basisShapes[i+1]
            i += 1

        l0 = _l0_proxy(normSum, mode=mode, T=T)
        sigma.append(l0)

    if debug:
        print("nSubspaces: ", nCumulativeSubspaces)
        print("Used:", nSubspacesUsedSoft)

    nCumulativeSubspaces = np.cumsum(nCumulativeSubspaces)

    lam = torch.tensor(0.0, device=layer.weights.device)
    for level in range(1, len(sigma)):
        multiplier = nCumulativeSubspaces[level - 1] + nSubspacesUsedSoft[level]
        lam = lam * (1 - sigma[level]) + multiplier * sigma[level]

    return lam

