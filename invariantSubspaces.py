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
    """
    No transformation applied to Wmat.
    """
    return Wmat

################# Reynolds operator for groups over images ###################
def G_rotation(Wmat):
    """
     Apply the Reynolds operator of the rotation group: {T_0, T_90, T_180, T_270} to Wmat
        (average over all the rotation transformations).
    :param Wmat: of shape (channels, height, width) or (height, width)
    :return: Matrix of same shape as Wmat transformed by the Reynolds operator of rotation group.
    """
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
    """
     Apply the Reynolds operator of the RGB channel permutation group to Wmat.
    :param Wmat: of shape (channels, height, width)
    :return: Transformed matrix of same shape as Wmat.
    """
    assert Wmat.shape[1] == Wmat.shape[2]
    nChannels = Wmat.shape[0]
    return Wmat.mean(axis=0, keepdims=True).repeat(nChannels, axis=0)


def G_horizontalFlip(Wmat):
    """
     Apply the Reynolds operator of the horizontal flip group: {e, T_h} to Wmat.
    :param Wmat: of shape (channels, height, width) or (height, width)
    :return: Transformed matrix of same shape as Wmat.
    """
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]
    return (Wmat + np.flip(Wmat, axis=axes[1])) / 2


def G_verticalFlip(Wmat):
    """
     Apply the Reynolds operator of the vertical flip group: {e, T_v} to Wmat.
    :param Wmat: of shape (channels, height, width) or (height, width)
    :return: Transformed matrix of same shape as Wmat.
    """
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]

    return (Wmat + np.flip(Wmat, axis=axes[0])) / 2


def G_flip(Wmat):
    """
     Apply the Reynolds operator of the flip group: {e, T_h, T_v, T_180} to Wmat.
        (have to include T_180 to make it a group).
    :param Wmat: of shape (channels, height, width) or (height, width)
    :return: Transformed matrix of same shape as Wmat.
    """
    if len(Wmat.shape) == 3:
        axes = (1,2)
    else:
        axes = (0,1)

    assert Wmat.shape[axes[0]] == Wmat.shape[axes[1]]
    return (Wmat + np.flip(Wmat, axis=axes[0])
            + np.flip(Wmat, axis=axes[1])
            + np.rot90(Wmat, k=2, axes=axes)) / 4

################################################################################


############## Reynolds operator for groups over sequences #####################
def G_permutation(Wmat, pos):
    """
     Apply the Reynolds operator of the Transposition group = {e, (i j)} to Wmat
        where  (i j) swaps Wmat[i] and Wmat[j]. Only the first dimension of Wmat is considered for swap.
    :param Wmat: of shape (sequence_length, sequence_dimension) for Sequences
    :param pos: Positions to swap: [i, j]
    :return: Matrix of same shape as Wmat after the transformation.
    """
    permut = list(range(Wmat.shape[0]))
    permut[pos[0]], permut[pos[1]] = permut[pos[1]], permut[pos[0]]
    return (Wmat + Wmat[permut]) / 2

################################################################################

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


################################### Invariant Subspaces ########################################
def projectAndRemove(A, B):
    """
    Given two matrices A and B (basis vectors in columns), remove the projection of A on B from A.
    Assumes that B is orthonormal.
    :return:
    """
    for i in range(A.shape[1]):
        A[:, i] -= (B @ B.T) @ A[:, i]
    return A


def constructTbar_onehot(i, transformation, Wshape):
    """
    Construct the matrix form of the Reynolds operator for a particular group.
    (via one-hot vectors)
    :param i: An integer denoting which column of the Reynolds operator matrix to construct
              (matrix is computed in parallel for all i).
    :param transformation: The function form of Reynolds operator of the group, e.g., G_rotation.
    :param Wshape: Shape of the input/weights. For instance for images, Wshape=(channels, kernel_size, kernel_size).
    :return: The i-th column of the matrix corresponding to the Reynolds operator of the given group.
    """
    rndW = np.zeros(np.prod(Wshape))
    rndW[i] = 1

    SrndW = transformation(rndW.reshape(*Wshape))
    return SrndW.reshape(np.prod(Wshape))

def getInvariantSubspace(Wshape, transformation):
    """
    Given the function form of the Reynolds operator of a group, e.g., G_rotation, obtain its left 1-eigenspace.
    :param Wshape: Shape of the input/weights.
                    For instance for images, Wshape=(channels, kernel_size, kernel_size).
                                 for sequences, Wshape=(sequence_length, dimension).
    :param transformation: The function form of Reynolds operator of the group, e.g., G_rotation.
    :return: Returns the left 1-eigenspace of shape (np.prod(Wshape), numEigenvectors),
                its complement and the eigenvalues.
    """

    Wsize = np.prod(Wshape)
    max_samples = Wsize

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = tmp.cpu_count()

    _log = utils.getLogger()
    _log.debug(f"Using {ncpus} CPUs to get {transformation.__name__}")

    pool = tmp.Pool(ncpus)

    ans = pool.map(
        partial(constructTbar_onehot, transformation=transformation, Wshape=Wshape),
        list(range(max_samples))
    )
    pool.close()
    pool.join()

    # Reynolds operator for the transformation
    Tbar = np.stack(ans, axis=1)

    # Eigenvectors of Tbar (symmetric)
    # Eigenvalues are in columns arranged in ascending order, 1-eigenvectors at the end.
    S, Vh = np.linalg.eigh(Tbar)

    rank = np.linalg.matrix_rank(np.diag(S), hermitian=True)

    # V: Eigenvectors associated with eigenvalue 1 and Vc is the complement space.
    V = Vh[:, -rank:]
    Vc = Vh[:, :-rank]

    return V, Vc, S

def showSubspace(subspace, Wshape, ndim=-1, channels=False):
    """
    Visualize a subspace.
    :param subspace: Subspace to visualize (for example obtained from `getInvariantSubspace` function).
                     Each column corresponds to a basis vector of the subspace.
    :param Wshape: Shape of the input/weights.
    :param ndim: Number of basis vectors to visualize
    :param channels: For images, visualize the channels separately or together as an RGB image.
    :return: None
    """
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
    """
    Find the intersection of subspaces U and V.
    :param U: Matrix with basis vectors of the subspace U in the columns.
    :param V: Matrix with basis vectors of the subspace V in the columns.
    :param method: Alternate methods for finding the intersection.
    :param decimals: Precision
    :return: Intersection subspace U \cap V.
    """
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
    """
    Find the intersection of a list of subspaces.
    :param listV: listV[i] contains a matrix corresponding to the subspace V_i.
    :param method: Alternate methods for finding the intersection.
    :param decimals: Precision
    :return: Intersection subspace V_0 \cap V_1 \ldots \cap V_{n-1}.
    """
    if method == 1:
        intersection = listV[0]
        for V in listV[1:]:
            intersection = _findIntersection(intersection, V, method=method, decimals=decimals)
    else:
        # intersection(A, B, ... ) = (join(A_c, B_c, ...))_c, where _c denotes complement.
        listVc = [complement(V) for V in listV]
        joinListVc = np.concatenate(listVc, axis=1)
        intersection = complement(joinListVc)
        intersection = np.around(intersection, decimals=decimals)

    return intersection

def complement(A):
    """
    Complement of a subspace A.
    :param A: Matrix with basis vectors of the subspace A in the columns.
    :return: Complement subspace of A.
    """
    return slinalg.null_space(A.T)

def _sortTransforms(listT):
    """
    Sort a given list of functions lexicographically by name
    (used to get unique filenames.)
    :param listT: List of functions (e.g., [G_rotation, G_flip]).
    :return: List lexicographically sorted.
    """
    return sorted(listT, key=lambda f: f.__name__)


def indexPowerSetGeneral(listT):
    """
    Obtain a generator to iterate over the power set of listT (i.e., the different subsets of listT) in the
    descending order of their sizes.
    :param listT: List of Reynolds operator transformations for various groups (e.g., [G_rotation, G_flip]).
    :return: A generator to iterate over the power set of listT (i.e., the different subsets of listT).
    """
    s = list(range(len(listT)))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s), -1, -1))

def indexPowerSetTranspositions(listT):
    """
    As with `indexPowerSetGeneral`, obtain a generator to iterate over the power set of listT, but assuming that
    listT contains the Reynolds operators of the \binom{n}{2} transposition groups (i.e., G_permutation for different
    values of (i j)). In this special case, we can iterate over the power set more efficiently as many of these subsets
    have null intersection and can be ignored.
    :param listT: List of Reynolds operator transformations for transposition groups.
    :return: A generator to iterate over the power set of listT (i.e., the different subsets of listT).
    """
    s = list(listT)
    yield list(range(len(s)))

    position = 0
    while position * (position - 1)//2 < len(listT):
        # Remove position
        currentPositions = []
        for gi,g in enumerate(listT):
            gName = g.keywords["pos"]
            if position not in gName:
                currentPositions.append(gi)

        yield(currentPositions)
        position += 1


def findAllSubspaces(listV, listT, powerSetMethod=None, method=2, decimals=6, debug=False):
    """
    Given a list of Reynolds operators of various groups and their respective left 1-eigenspaces, compute the
    full lattice of subspaces. See Theorem 3 or Algorithm 1 in the paper.

    :param listT: List of Reynolds operator transformations for various groups (e.g., [G_rotation, G_flip]).
    :param listV: For the Reynolds operator in listT[i], listV[i] is the corresponding 1-eigenspace already computed
                    in `getInvariantSubspace` function.
    :param powerSetMethod: Method to iterate over the power set of listT (i.e., the different subsets of listT).
    :param method: Alternate methods to compute subspace intersections
    :param decimals: Precision
    :param debug: Print logs
    :return: finalSubspaces: All the nonempty subspaces arranged in non-increasing order of invariance.
             finalPowerSetConfigs: All the subsets of listT (encoded by indices) with nonempty subspace in the lattice.
    """

    Wsize = listV[0].shape[0]

    if powerSetMethod is not None:
        # Efficient iteration over the power set for permutation groups.
        # Do not compute subspaces that are known to have empty intersection.
        indexPowerSetConfigs = powerSetMethod(listT)
    else:
        indexPowerSetConfigs = indexPowerSetGeneral(listT)

    complete = np.zeros((Wsize, 0))

    finalSubspaces = []
    finalPowerSetConfigs = []

    # Iterate over the power set (via indices of listT)
    # For example: configs=[0,1,...n-1] is the full set listT, whereas configs=[2,3] is the subset {listT[2], listT[2]}.
    # Note that len(configs) gives us the level in the lattice:
    #       - len(configs)=n is the topmost level with highest invariance.
    #       - len(configs)=0 is the bottommost level with no invariance.
    for configs in indexPowerSetConfigs:
        configs_ = set(configs)
        if debug:
            print("=" * 80)

        if complete.shape[1] > 0:
            complete = slinalg.orth(complete)

        if complete.shape[0] == complete.shape[1]:
            # Found the bases for the entire space. Exit the loop.
            break

        if listT is not None:
            print("Finding subspace of ", [listT[i].__name__ for i in configs_])

        # Find the intersection of all the subspaces in this particular subset (given by variable `configs`)
        # This computes \tilde{B}_M in Theorem 3 or Equation (8) in the paper.
        if len(configs) > 0:
            B = [listV[i] for i in configs_]
            intersection = findIntersection(B, method=method)
        else:
            intersection = np.eye(Wsize)

        # Remove orthogonal projections of the subspaces already computed (in higher levels).
        # This computes B_M in Theorem 3 or Equation (8) in the paper.
        supersetIndices = [i for i, N in enumerate(finalPowerSetConfigs) if configs_.issubset(N)]
        if len(supersetIndices) > 0:
            supersetSubspaces = np.concatenate([finalSubspaces[i] for i in supersetIndices],  axis=1)
            intersection = projectAndRemove(intersection, slinalg.orth(supersetSubspaces))

        # If the subspace is empty, ignore and continue.
        if np.allclose(intersection, 0, atol=10**(-decimals)):
            continue

        # Make it orthonormal
        intersection = slinalg.orth(intersection)

        # Add the subspace to the lattice of subspaces.
        finalPowerSetConfigs.append(configs)
        finalSubspaces.append(intersection)

        # Also, append the subspace with all the subspaces computed till now.
        complete = np.concatenate([complete, intersection], axis=1)

        if debug:
            print(f"Found subspace of size: {intersection.shape}")
            print(intersection.round(decimals=6))
            print(complete.round(decimals=6))
    return finalSubspaces, finalPowerSetConfigs


def findAllSubspacesFromTransforms(listT, Wshape, powerSetMethod=None, method=2, decimals=6, show=False):
    """
    Given a list of Reynolds operators of various groups, computes their respective 1-eigenspaces
    and calls `findAllSubspaces` function to find the full lattice of subspaces.

    :param listT: List of Reynolds operator transformations for various groups (e.g., [G_rotation, G_flip]).
    :param Wshape: Shape of input/weights.
    :param powerSetMethod: Method to iterate over the power set of listT (i.e., the different subsets of listT).
    :param method: Alternate methods to compute subspace intersections
    :param decimals: Precision
    :param show: Show the subspaces for debugging purposes.
    :return: allSubspaces: All the nonempty subspaces arranged in non-increasing order of invariance.
             powerSetConfigs: All the subsets of listT (encoded by indices) with nonempty subspace in the lattice.
    """

    _log = utils.getLogger()
    listT = _sortTransforms(listT)

    # For every i, construct 1-eigenspace for the Reynolds operator listT[i].
    listV = []
    for transform in listT:
        A, Ac, S = getInvariantSubspace(Wshape, transform)
        listV.append(A)
        if show:
            _log.debug(f"Invariant subspace of transform {transform.__name__}: {A.shape}")
            showSubspace(A, Wshape, 1, channels=True)

    allSubspaces, powerSetConfigs = findAllSubspaces(listV=listV, listT=listT, powerSetMethod=powerSetMethod, method=method, decimals=decimals)

    if show:
        for subspace, config in zip(allSubspaces, powerSetConfigs):
            showSubspace(subspace, Wshape, 1, channels=True)

    return allSubspaces, powerSetConfigs


def _getBasisFileName(listT, Wshape):
    """
    Unique filename to store the basis
    :param listT: Reynolds operators (as functions)
    :param Wshape: Shape of input/weights
    :return: Filename
    """
    listT = _sortTransforms(listT)
    transformNames = [t.__name__ for t in listT]
    transformString = "transforms=[" + ",".join(transformNames) + "]"

    if len(transformString) > 100:
        transformString = sha1(transformString.encode()).hexdigest()

    Wstring = "Wshape=(" + ",".join([str(ws) for ws in Wshape]) + ")"
    fileName = transformString + "|" + Wstring + ".npz"

    return fileName

def saveAllBasis(folder, listT, Wshape, allSubspaces, powerSetConfigs):
    """
    Save basis for the lattice of subspaces in given folder.
    :param folder: Folder to save
    :param listT: List of Reynolds operators (functions)
    :param Wshape: Shape of input/weights.
    :param allSubspaces: Lattice of subspaces
    :param powerSetConfigs: The subsets of listT (encoded by indices) that have nonempty subspaces.
    :return: Saved file name
    """
    listT = _sortTransforms(listT)
    fileName = _getBasisFileName(listT, Wshape)

    pcString = ["_".join([str(pci) for pci in pc]) for pc in powerSetConfigs]
    kwargs = dict(zip(pcString, allSubspaces))

    np.savez(f"{os.path.join(folder, fileName)}", **kwargs)
    return fileName

def loadAllBasis(folder, listT, Wshape):
    """
    Load basis from file
    :param folder: Basis folder
    :param listT: List of Reynolds operators (functions)
    :param Wshape: Shape of input/weights
    :return: listT: Lexicographically sorted list of Reynolds operators (functions)
             allSubspaces: Lattice of subspaces
             powerSetConfigs: The subsets of listT (encoded by indices) that have nonempty subspaces.
    """
    listT = _sortTransforms(listT)
    fileName = _getBasisFileName(listT, Wshape)
    loadedBasis = np.load(f"{os.path.join(folder, fileName)}")
    powerSetConfigs = loadedBasis.files
    allSubspaces = [loadedBasis[key] for key in loadedBasis.files]

    return listT, allSubspaces, powerSetConfigs

def getAllBasis(folder, listT, Wshape, powerSetMethod=None, force=False):
    """
    Loads from file. Creates all subspaces if file does not exists.
    :param folder: Folder to load from or save to.
    :param listT: List of Reynolds operators (functions)
    :param Wshape: Shape of input/weights
    :param powerSetMethod: Method to iterate over the power set of listT (i.e., the different subsets of listT).
    :param force: Force re-create all subspaces.
    :return: Same as `loadAllBasis` function.
             listT: Lexicographically sorted list of Reynolds operators (functions)
             allSubspaces: Lattice of subspaces
             powerSetConfigs: The subsets of listT (encoded by indices) that have nonempty subspaces.
    """
    _log = utils.getLogger()
    try:
        assert force==False
        basis = loadAllBasis(folder, listT, Wshape)
        return basis
    except:
        _log.info(f"Failed to load basis file. Creating basis file at {folder}.")
        allSubspaces, powerSetConfigs = findAllSubspacesFromTransforms(listT=listT, Wshape=Wshape, powerSetMethod=powerSetMethod, show=False)
        fileName = saveAllBasis(folder=folder, listT=listT, Wshape=Wshape, allSubspaces=allSubspaces, powerSetConfigs=powerSetConfigs)

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
        listT, allSubspaces, powerSetConfigs = getAllBasis(folder=basisDir, listT=listT, Wshape=Wshape,
                                                       powerSetMethod=indexPowerSetTranspositions, force=True)
    else:
        basisDir = os.path.join(dataDir, 'basis')
        os.makedirs(basisDir, exist_ok=True)
        inputChannels = 3
        kernelSize = (3, 3)
        Wshape = (inputChannels, kernelSize[0], kernelSize[1])
        listT = [G_rotation, G_color_permutation, G_verticalFlip]
        listT, allSubspaces, powerSetConfigs = getAllBasis(folder=basisDir, listT=listT, Wshape=Wshape, force=True)

