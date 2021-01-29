import numpy as np
import scipy.linalg as slinalg
import torch.multiprocessing as tmp
import os
import itertools
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Common import utils
from hashlib import sha1

def G_trivial(Wmat):
    # Trivial group
    return Wmat

def G_permutation(Wmat, pos):
    # Transposition group = {e, (i j)}
    permut = list(range(Wmat.shape[0]))
    permut[pos[0]], permut[pos[1]] = permut[pos[1]], permut[pos[0]]
    return (Wmat + Wmat[permut]) / 2


def G_rotation(Wmat):
    # Rotation group: {T_0, T_90, T_180, T_270}
    # Wmat can be either (C, H, W) or (H, W)
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]

    finalWmat = np.zeros_like(Wmat)
    count = 0
    for rotationDeg in [0, 90, 180, 270]:
        finalWmat += np.rot90(Wmat, k=rotationDeg//90, axes=axes)
        count += 1

    return finalWmat / count


def G_color_permutation(Wmat):
    # Color permutation group
    # Wmat has to be (C, H, W)
    assert Wmat.shape[1] == Wmat.shape[2]
    nChannels = Wmat.shape[0]
    return Wmat.mean(axis=0, keepdims=True).repeat(nChannels, axis=0)


def G_horizontalFlip(Wmat):
    # Horizontal flip group: {e, T_h}
    # Wmat can be either (C, H, W) or (H, W)
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]
    return (Wmat + np.flip(Wmat, axis=axes[1])) / 2


def G_verticalFlip(Wmat):
    # Vertical flip group: {e, T_v}
    # Wmat can be either (C, H, W) or (H, W)
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]

    return (Wmat + np.flip(Wmat, axis=axes[0])) / 2


def G_flip(Wmat):
    # Flip group: {e, T_h, T_v, T_180}  (Have to include T_180 to make it a group).
    # Wmat can be either (C, H, W) or (H, W)
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]
    return (Wmat + np.flip(Wmat, axis=axes[0])
            + np.flip(Wmat, axis=axes[1])
            + np.rot90(Wmat, k=2, axes=axes)) / 4


def getTransformationsFromNames(names):
    mappingDict = {
        "rotation": G_rotation,
        "horizontal_flip": G_horizontalFlip,
        "vertical_flip": G_verticalFlip,
        "flip": G_flip,
        "color_permutation": G_color_permutation,
        "trivial": G_trivial,
    }

    return [mappingDict[name.lower()] for name in names]

def projectAndRemove(A, B):
    # Remove projection of A on B from A.
    # Assumes B is orthonormal.
    for i in range(A.shape[1]):
        A[:, i] -= (B @ B.T) @ A[:, i]
    return A


def constructTbar_rand(i, transformation, Wshape):
    # Construct the Reynolds operator for the group (given by `transformation` function, e.g. G_rotation)
    radius = np.random.uniform(0, 1, size=1)
    rndW = radius * np.random.normal(0, 1, size=Wshape)
    SrndW = transformation(rndW)
    return SrndW.reshape(np.prod(Wshape))

def constructTbar_onehot(i, transformation, Wshape):
    rndW = np.zeros(np.prod(Wshape))
    rndW[i] = 1

    SrndW = transformation(rndW.reshape(*Wshape))
    return SrndW.reshape(np.prod(Wshape))

def getInvariantSubspace(Wshape, transformation, method="onehot"):
    Wsize = np.prod(Wshape)
    if method == "rand":
        max_samples = 100 * Wsize
    else:
        max_samples = Wsize

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = tmp.cpu_count()

    _log = utils.getLogger()
    _log.debug(f"Using {ncpus} CPUs to get {transformation.__name__}")

    pool = tmp.Pool(ncpus)

    if method == "rand":
        ans = pool.map(
            partial(constructTbar_rand, transformation=transformation, Wshape=Wshape),
            list(range(max_samples)),
        )
    else:
        ans = pool.map(
            partial(constructTbar_onehot, transformation=transformation, Wshape=Wshape),
            list(range(max_samples))
        )
    pool.close()
    pool.join()

    # Reynolds operator for the transformation
    Tbar = np.stack(ans, axis=1)

    # Eigenvectors
    # Tbar symmetric
    _, S, Vh = np.linalg.svd(Tbar, hermitian=True)
    rank = np.linalg.matrix_rank(np.diag(S), hermitian=True)

    # V: Eigenvectors associated with eigenvalue 1
    V = Vh[:rank, :]
    Vc = Vh[rank:, :]

    return V.T, Vc.T, S

def showSubspace(subspace, Wshape, ndim=-1, channels=False):
    subspace = subspace.T

    if ndim == -1:
        ndim = subspace.shape[0]
    subspace = subspace[:ndim]

    ndim = subspace.shape[0]
    maxCols = min(ndim, 4)

    for j in range(ndim):
        if j % maxCols == 0:
            plt.show()
            nCols = maxCols if ndim - j > maxCols else ndim - j
            fig, axes = plt.subplots(1, nCols, figsize=(12 * nCols // 2, 9 // 2))
            try:
                axes[0]
            except:
                axes = [axes]

        kernel = subspace[j]
        kernel = kernel.reshape(*Wshape)

        if len(kernel.shape) == 3:
            kernel = kernel.transpose(1, 2, 0)
            if channels:
                kernel = np.concatenate([kernel[:, :, c] for c in range(kernel.shape[-1])], axis=1)
                axes[j%maxCols].add_patch(patches.Rectangle((-0.45, -0.45), 2.95, 2.95, facecolor='none', linestyle='--', linewidth=2, edgecolor='tab:red'))
                axes[j%maxCols].add_patch(patches.Rectangle((2.55, -0.45), 2.95, 2.95, facecolor='none', linestyle='--', linewidth=2, edgecolor='tab:green'))
                axes[j%maxCols].add_patch(patches.Rectangle((5.55, -0.45), 2.95, 2.95, facecolor='none', linestyle='--', linewidth=2, edgecolor='tab:blue'))

        axes[j%maxCols].imshow(kernel.round(decimals=6), cmap="Greys")
        axes[j%maxCols].set_xticklabels([0] + [0, 1, 2] * 3)

    plt.show()


def _findIntersection(U, V, method=1, decimals=6):
    # Columns of vectors in U and V: column space defines the subspace.
    if method == 1:
        X = np.concatenate([U, V], axis=1)
        coeffs = slinalg.null_space(X)
        coeffs = coeffs[:U.shape[1]]
        ans = U @ coeffs
    else:
        nullspace_Ut = slinalg.null_space(U.T)
        nullspace_Vt = slinalg.null_space(V.T)

        nullspace_UVt = np.concatenate([nullspace_Ut, nullspace_Vt], axis=1)
        ans = slinalg.null_space(nullspace_UVt.T)

    ans = np.around(ans, decimals=decimals)
    return ans


def findIntersection(listV, method=1, decimals=6):
    # Columns of vectors all V in listV: column space defines the subspace.
    if method == 1:
        intersection = listV[0]
        for V in listV[1:]:
            intersection = _findIntersection(intersection, V, method=method, decimals=decimals)
    else:
        # intersection(A, B, ... ) = (join(A_c, B_c, ...))_c
        listVc = [complement(V) for V in listV]
        joinListVc = np.concatenate(listVc, axis=1)
        intersection = complement(joinListVc)
        intersection = np.around(intersection, decimals=decimals)

    return intersection

def complement(A):
    return slinalg.null_space(A.T)

def _sortTransforms(listT):
    return sorted(listT, key=lambda f: f.__name__)


def indexPowerSetGeneral(groups):
    s = list(range(len(groups)))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s), -1, -1))

def indexPowerSetTranspositions(groups):
    s = list(groups)
    yield list(range(len(s)))

    position = 0
    while position * (position - 1)//2 < len(groups):
        # Remove position
        currentPositions = []
        for gi,g in enumerate(groups):
            gName = g.keywords["pos"]
            if position not in gName:
                currentPositions.append(gi)

        yield(currentPositions)
        position += 1


def findAllSubspaces(listV, listT, powerSetMethod=None, method=2, decimals=6, debug=False):
    # Lists of Invariant subspaces.

    Wsize = listV[0].shape[0]
    nSubspaces = len(listV)

    if powerSetMethod is not None:
        # Efficient computation for the power set
        # Do not compute those subspaces that are known to have empty intersection.
        indexPowerSetConfigs = powerSetMethod(listT)
    else:
        indexPowerSetConfigs = indexPowerSetGeneral(listT)

    previous = np.zeros((Wsize, 0))
    current = np.zeros((Wsize, 0))

    finalPowerSets = []
    finalPowerSetConfigs = []

    currentLevel = nSubspaces
    previousLevel = currentLevel

    levelBased = True

    for configs in indexPowerSetConfigs:
        if debug:
            print("=" * 80)
        currentLevel = len(configs)
        if levelBased:
            check = np.concatenate([previous, current], axis=1)
            if check.shape[1] > 0:
                check = slinalg.orth(check)

            if check.shape[0] == check.shape[1]:
                # Found all bases
                break

            if currentLevel < previousLevel:
                if debug:
                    print(f"Level change from {previousLevel} --> {currentLevel}")
                previous = check
                current = np.zeros((Wsize, 0))
                previousLevel = currentLevel

        if listT is not None:
            print("Finding subspace of ", [listT[i].__name__ for i in configs])

        if currentLevel > 0:
            B = [listV[i] for i in configs]
            intersection = findIntersection(B, method=method)
        else:
            intersection = np.eye(Wsize)

        if previous.shape[1] > 0:
            intersection = projectAndRemove(intersection, previous)

        if np.allclose(intersection, 0, atol=10**(-decimals)):
            continue

        intersection = slinalg.orth(intersection)

        finalPowerSetConfigs.append(configs)
        finalPowerSets.append(intersection)

        # Add to the current level
        if levelBased:
            current = np.concatenate([current, intersection], axis=1)
        else:
            previous = np.concatenate([previous, intersection], axis=1)
            previous = slinalg.orth(previous)

            if previous.shape[0] == previous.shape[1]:
                # Found all bases
                break

        if debug:
            print(intersection.round(decimals=6))
            print(previous.round(decimals=6))
    return finalPowerSets, finalPowerSetConfigs


def findAllSubspacesFromTransforms(listT, Wshape, powerSetMethod=None, method=2, decimals=6, show=False):
    _log = utils.getLogger()
    listT = _sortTransforms(listT)
    listV = []
    for transform in listT:
        A, Ac, S = getInvariantSubspace(Wshape, transform)
        listV.append(A)
        if show:
            _log.debug(f"Invariant subspace of transform {transform.__name__}: {A.shape}")
            showSubspace(A, Wshape, 1, channels=True)

    powerSet, powerSetConfigs = findAllSubspaces(listV=listV, listT=listT, powerSetMethod=powerSetMethod, method=method, decimals=decimals)

    if show:
        for subspace, config in zip(powerSet, powerSetConfigs):
            ppp = [f"{t.__name__} = {c}" for c, t in zip(config, listT)]
            _log.debug(", ".join(ppp))
            showSubspace(subspace, Wshape, 1, channels=True)

    return powerSet, powerSetConfigs


def _getBasisFileName(listT, Wshape):
    listT = _sortTransforms(listT)
    transformNames = [t.__name__ for t in listT]
    transformString = "transforms=[" + ",".join(transformNames) + "]"

    if len(transformString) > 100:
        transformString = sha1(transformString.encode()).hexdigest()

    Wstring = "Wshape=(" + ",".join([str(ws) for ws in Wshape]) + ")"
    fileName = transformString + "|" + Wstring + ".npz"

    return fileName

def saveAllBasis(folder, listT, Wshape, powerSet, powerSetConfigs):
    listT = _sortTransforms(listT)
    fileName = _getBasisFileName(listT, Wshape)

    pcString = ["_".join([str(pci) for pci in pc]) for pc in powerSetConfigs]
    kwargs = dict(zip(pcString, powerSet))

    np.savez(f"{os.path.join(folder, fileName)}", **kwargs)
    return fileName

def loadAllBasis(folder, listT, Wshape):
    listT = _sortTransforms(listT)
    fileName = _getBasisFileName(listT, Wshape)
    powerSetz = np.load(f"{os.path.join(folder, fileName)}")
    powerSetConfigs = powerSetz.files
    powerSet = [powerSetz[key] for key in powerSetz.files]

    return listT, powerSet, powerSetConfigs

def getAllBasis(folder, listT, Wshape, powerSetMethod=None, force=False):
    # Loads from file. Creates if file does not exists.
    _log = utils.getLogger()
    try:
        assert force==False
        basis = loadAllBasis(folder, listT, Wshape)
        return basis
    except:
        _log.info(f"Failed to load basis file. Creating basis file at {folder}.")
        powerSet, powerSetConfigs = findAllSubspacesFromTransforms(listT=listT, Wshape=Wshape, powerSetMethod=powerSetMethod, show=False)
        fileName = saveAllBasis(folder=folder, listT=listT, Wshape=Wshape, powerSet=powerSet, powerSetConfigs=powerSetConfigs)

        _log.info(f"Saved basis file: {os.path.join(folder, fileName)}.")

    return loadAllBasis(folder, listT, Wshape)


if __name__ == '__main__':
    # Usage
    dataDir = "tmp"     # Storing basis in a tmp directory

    inputType = "image"
    if inputType == "sequence":
        basisDir = os.path.join(dataDir, 'basis')
        os.makedirs(basisDir, exist_ok=True)
        n = 5       # Sequence length
        Wshape = (n, 1)
        listT = []
        for ij in list(itertools.combinations(range(n), 2)):
            f = partial(G_permutation, pos=ij)
            f.__name__ = f"({ij[0]} {ij[1]})"
            listT.append(f)
        listT, powerSet, powerSetConfigs = getAllBasis(folder=basisDir, listT=listT, Wshape=Wshape,
                                                       powerSetMethod=indexPowerSetTranspositions, force=False)
    else:
        basisDir = os.path.join(dataDir, 'basis')
        os.makedirs(basisDir, exist_ok=True)
        inputChannels = 3
        kernelSize = (3, 3)
        Wshape = (inputChannels, kernelSize[0], kernelSize[1])
        listT = [G_rotation, G_color_permutation, G_flip]
        listT, powerSet, powerSetConfigs = getAllBasis(folder=basisDir, listT=listT, Wshape=Wshape, force=False)

