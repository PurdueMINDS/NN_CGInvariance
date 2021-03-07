import numpy as np
import torch


def invariancePenalty(layer, mode=None, T=None, debug=False):
    """
    Returns the penalty for the layer depending on the subspaces used.
    A subspace is "used" if the corresponding coefficients/weights in the layer are non-zero.
    1. Find the least invariant subspace used, say, B.
    2. The first term in the penalty counts the number of subspaces (used or unused) that are invariant to more groups than B.
       This term ensures that the optimization tries to use subspaces that are higher in the partial order with
       invariance to more groups.
    3. The second term in the penalty counts the number of subspaces that have the same level of invariance as B, and
       are also "used". The larger the second term, farther away the optimization is from increasing the least level of
       invariance.
    This function returns a differentiable approximation of this penalty.

    :param layer: A CG-invariant layer; can be Images.models.CGConv2D or Sequences.models.CGSequenceLayer
    :param mode: Different approximations of the L0 norm.
    :param T: Temperature for the L0 approximation.
    :param debug: Debug print statements
    :return: Penalty
    """

    def _l0_proxy(w, mode, T):
        # Different approximations of the L0 norm
        if mode is None or mode.lower() == "true":
            return w.norm(p=0)
        elif mode.lower() == "simple":
            return T * w / (T * w + 1)
        elif mode.lower() == "sigmoid":
            return 2 * torch.sigmoid(T * w) - 1
        else:
            raise NotImplementedError

    # For all the subspaces, find the level of their invariance, i.e., the number of groups
    #       each subspace is invariant to.
    invarianceOfSubspaces = [
        len(config.split("_")) if config != "" else 0 for config in layer.basisConfigs
    ]

    i = -1
    prevBasisShape = 0
    nCumulativeSubspaces = []       # Cumulative number of subspaces till level i.
    nSubspacesUsed = []             # Tracks the number of subspaces "used" in each level.
    nSubspacesUsedSoft = []         # Soft version of nSubspacesUsed.
    beta = []                       # Tracks whether i-th invariance level is "used" or not.

    while i < len(invarianceOfSubspaces) - 1:
        # We will edit these values in the loop.
        nCumulativeSubspaces.append(0)
        nSubspacesUsed.append(0)
        nSubspacesUsedSoft.append(0)

        # The current invariance level
        #   (i.e., the number of groups the subspaces are invariant to in the current level of the lattice).
        current = invarianceOfSubspaces[i + 1]
        normSum = 0

        # Keep iterating till we are at the same level `current`.
        while (
            i < len(invarianceOfSubspaces) - 1
            and invarianceOfSubspaces[i + 1] == current
        ):

            # Obtain the index of the coefficients/weights corresponding to the basis of
            #   the current subspace.
            index = torch.arange(
                prevBasisShape,
                prevBasisShape + layer.basisShapes[i + 1],
                device=layer.weights.device,
            )

            # Find the L2 norm of the coefficients corresponding to this subspace.
            normSumi = layer.weights.index_select(dim=-2, index=index).norm(p=2)

            # Add the norm to the sum (which computes L2 norm of coefficients for the entire invariance level).
            normSum += normSumi

            # Add 1 to the subspace counter of the current invariance level.
            nCumulativeSubspaces[-1] += 1

            # Add 1 to the "used" subspace counter of the current invariance level IF the L2 norm of coefficients of
            #       computed > 0. This is only used for debug.
            if normSumi > 0:
                nSubspacesUsed[-1] += 1

            # Use a differentiable approximation of the if-condition above using a proxy for L0 of `normSumi`.
            # This corresponds to a soft version of the nSubspacesUsed counter.
            nSubspacesUsedSoft[-1] += _l0_proxy(normSumi, mode=mode, T=T)

            # Move the prevBasis pointer so that we can index the next one appropriately.
            prevBasisShape += layer.basisShapes[i + 1]
            i += 1

        # Obtain \beta_l for all levels l, once again using the L0 approximation.
        # \beta_l approximates if at least one subspace of the current level was "used".
        l0 = _l0_proxy(normSum, mode=mode, T=T)
        beta.append(l0)

    if debug:
        print("nSubspaces: ", nCumulativeSubspaces)
        print("Used:", nSubspacesUsedSoft)

    # Find cumulative sum of subspace counter: to get the number of subspaces in all levels above level l.
    nCumulativeSubspaces = np.cumsum(nCumulativeSubspaces)

    # Recursive formula to find R(\bOmega); see Appendix F.2 (Differentiable Approximation) in the paper.
    # Variables and paper notation:- R: R(\bOmega),  f_level: f_l(\bOmega), beta[level]: \beta_l(\bOmega)
    R = torch.tensor(0.0, device=layer.weights.device)
    for level in range(1, len(beta)):
        # f_level gives the two terms of the penalty.
        #   First term is the number of subspaces above the level.
        #   Second term is the number of subspaces "used" in the level
        # However, we need the recursive procedure because the penalty is defined for the least invariance level used.

        f_level = nCumulativeSubspaces[level - 1] + nSubspacesUsedSoft[level]
        R = R * (1 - beta[level]) + f_level * beta[level]

    # R is the final penalty
    return R
