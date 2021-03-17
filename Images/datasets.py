import torch
from torchvision import transforms
import numpy as np
import os
import itertools
from torchvision.datasets import VisionDataset, MNIST
from PIL import Image
import matplotlib.pyplot as plt
from Common import utils
from functools import partial


class ExtrapolationDataset(VisionDataset):
    """
    Extrapolation dataset for MNIST
    """

    train_file = "train.pt"
    test_file = "test.pt"

    def __init__(
        self,
        root,
        name,
        groups,
        mode,
        train=True,
        fold=None,
        pSamples=None,
        seed=42,
        transform=None,
        target_transform=None,
        normalize_transform=None,
        download=False,
        force=False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # :mode:
        # :fold: (i,k) to indicate the fold i of a k-fold cross validation; None for no CV.
        # :pSamples: Percentage to sample. Used to obtain the effect of training dataset size. Use None for no subsampling.
        # :root, train, transform, target_transform, download: Standard parameters
        # :normalize_transform: Do not pass Normalize() in transform.
        #        Pass True for trainData (automatically computed). Use trainData's normalizeTransform for testData.
        # :force: Force re-create dataset.

        self.root = root
        self.name = name.upper() + "Xtra"       # Name of the dataset, for example MNISTXtra
        self.train = train
        self.normalize_transform = normalize_transform
        self.force = force

        # Groups used to transform the data.
        # For example: [G_rotation_create, G_flip_create]
        self.groups = groups

        # Codes which of the groups are in G_I and which are in G_D
        self.mode = mode

        if download or force:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            self.data_file = self.train_file
            splitPrefix = "train"
        else:
            self.data_file = self.test_file
            splitPrefix = "test"

        self.data, self.targets = torch.load(os.path.join(self.folder, self.data_file))

        _log = utils.getLogger()

        # Cross validation folds
        if fold is not None:
            cvIt = fold[0]
            cvFolds = fold[1]

            self.split_file = f"{splitPrefix}_cvFolds={cvFolds}_seed={seed}.pt"
            if not self._check_exists(self.split_file):
                _log.info("Split file does not exist. Creating one.")
                splits = randomKFoldSplit(len(self.data), cvFolds, seed)
                torch.save(splits, os.path.join(self.folder, self.split_file))

            # _log.info(f"Using splits from {self.split_file}")
            splits = torch.load(os.path.join(self.folder, self.split_file))

            allExceptFold = np.concatenate(splits[:cvIt] + splits[cvIt + 1 :])

            self.data = self.data[allExceptFold]
            self.targets = self.targets[allExceptFold]

        classes = self.targets.unique().sort().values
        self.class_to_idx = {
            class_name.item(): i for i, class_name in enumerate(classes)
        }

        if pSamples is not None:
            # Percentage to sample : To find effect of dataset size.
            N = self.data.shape[0]
            sampleIdx = np.random.choice(N, int(N * pSamples / 100), replace=False)
            self.data = self.data[sampleIdx]
            self.targets = self.targets[sampleIdx]

        # Compute input normalization (in the training data).
        if self.train and self.normalize_transform:
            mean = self.data.float().mean(dim=(0, 1, 2)) / 255
            std = self.data.float().std(dim=(0, 1, 2)) / 255
            std[std == 0] = 1
            self.normalize_transform = transforms.Normalize(mean, std)

    def __getitem__(self, index):
        img, target = (
            self.data[index],
            int(self.class_to_idx[self.targets[index].item()]),
        )

        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.normalize_transform:
            img = self.normalize_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def folder(self):
        groupFolderName = ":".join([g_create.__name__[:-7] for g_create in self.groups])
        return os.path.join(self.root, self.name, groupFolderName, self.mode)

    def _check_exists(self, file=None):
        if file is None:
            return os.path.exists(
                os.path.join(self.folder, self.train_file)
            ) and os.path.exists(os.path.join(self.folder, self.test_file))
        else:
            return os.path.exists(os.path.join(self.folder, file))

    def download(self):
        # Actually download MNIST and create the individual datasets.
        _log = utils.getLogger()
        if not self.force and self._check_exists():
            return

        os.makedirs(self.folder, exist_ok=True)

        if self.name.lower() == "mnistxtra":
            nChannels = 1
            subsetLabels = [3, 4]
            datasetClass = MNIST
        elif self.name.lower() == "mnistfullxtra":
            nChannels = 1
            subsetLabels = None
            datasetClass = MNIST
        else:
            raise NotImplementedError

        trainData = datasetClass(self.root, train=True, download=True)

        if type(trainData.data) != torch.Tensor:
            trainData.data = torch.tensor(trainData.data)

        if type(trainData.targets) != torch.Tensor:
            trainData.targets = torch.tensor(trainData.targets)

        testData = datasetClass(self.root, train=False)
        if type(testData.data) != torch.Tensor:
            testData.data = torch.tensor(testData.data)

        if type(testData.targets) != torch.Tensor:
            testData.targets = torch.tensor(testData.targets)

        _log.info(
            f"Processing {datasetClass.__name__} dataset to create {self.name} [{self.mode}] dataset."
        )
        np.random.seed(42)
        # Modes
        modeList = self.mode.split("_")

        color = (255, 0, 0)
        try:
            labelType = modeList[-1]
        except:
            labelType = "replace"

        # Use only a subset of digits
        if subsetLabels:
            subset(trainData, subsetLabels)
            subset(testData, subsetLabels)

        # MNIST Grayscale to RGB (color all digits red)
        if nChannels == 1:
            makeThreeChannels(trainData, color=color)
            makeThreeChannels(testData, color=color)

        # Hypothesis 1 : Selectively able to choose the group
        config = modeList[1]

        # Groups the label depends on (given by `mode`)
        G_D = [
            self.groups[i].__name__
            for i in range(len(config))
            if config[i] == "1"
        ]

        # Groups the label is invariant to (given by `mode`)
        G_I = [
            self.groups[i].__name__
            for i in range(len(config))
            if config[i] == "0"
        ]
        print(f"G_D: {G_D}, G_I: {G_I}")
        for i, g_create in enumerate(self.groups):
            if config[i] == "1":
                # config[i] = 1: The group affects the label.
                # Apply the group transformations to both training and test data, and change labels accordingly.
                randomGroupTransformation(trainData, g_create, changeTarget=labelType)
                randomGroupTransformation(testData, g_create, changeTarget=labelType)
                if labelType == "replace":
                    labelType = "add"  # "add" the second label
            else:
                # config[i] = 0: The group DOES NOT affect the label.
                # Apply the group transformations only to the test data.
                # Training data need not see these transformations (economic sampling).
                randomGroupTransformation(testData, g_create)

        torch.save(
            [trainData.data, trainData.targets],
            os.path.join(self.folder, self.train_file),
        )
        torch.save(
            [testData.data, testData.targets], os.path.join(self.folder, self.test_file)
        )

        _log.info("Done creating dataset.")


#### Utilities ####
def randomKFoldSplit(items, num_folds, seed=None):
    # Generate a k-fold random split.
    np.random.seed(seed)
    indices = np.random.permutation(items)
    splits = np.array_split(indices, num_folds)
    for fold in splits:
        fold.sort()
    return splits


def showAll(data, nplots=10, title=None):
    # Visualize the data
    X = data.data
    y = data.targets
    idx = np.random.choice(X.shape[0], nplots, replace=False)
    X = X[idx]
    y = y[idx]

    nrows = nplots // 3
    ncols = 3
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12 // 2 * ncols, 9 // 2 * nrows)
    )

    for r in range(nrows):
        for c in range(ncols):
            numberValue = 3 * r + c
            number = X[numberValue]
            axes[r][c].imshow(number, cmap="Greys")
            axes[r][c].set_title(f"Label = {y[numberValue].item()}")

    if title:
        plt.suptitle(title, fontsize=20)
    plt.show()


def subset(visionData, labels):
    # Obtain a subset of MNIST (for example with only 3s and 4s).
    data = visionData.data
    targets = visionData.targets

    idx = np.array([], dtype="int64")
    for label in labels:
        idxi = np.where(targets == label)[0]
        idx = np.concatenate([idx, idxi])

    visionData.data = data[idx]
    visionData.targets = targets[idx]


def makeThreeChannels(visionData, color=(255, 255, 255)):
    # Gray scale to RGB.
    if len(visionData.data.shape) == 3:
        visionData.data = visionData.data.unsqueeze(3).repeat([1, 1, 1, 3])

    threshold = 50  # To avoid small noises being pushed to 255.
    mask = (visionData.data > threshold).any(dim=-1)
    for i in range(3):
        visionData.data[:, :, :, i][mask] = color[i]
        visionData.data[:, :, :, i][~mask] = 0


def G_rotation_create(Xshape):
    # Randomly rotate an image by some multiple of 90-degree.
    group = list(range(4))

    def action(X, gIdx):
        return np.rot90(X, k=group[gIdx], axes=(0, 1))

    return len(group), action


def G_channelPermutation_create(Xshape, mode=None):
    # Randomly permute the channels of an image.
    if mode == "circular":
        arr = np.arange(Xshape[-1])
        group = [np.roll(arr, i) for i in range(Xshape[-1])]
    else:
        group = list(itertools.permutations(range(Xshape[-1])))

    def action(X, gIdx):
        return X[:, :, group[gIdx]]

    return len(group), action


G_channelCircularPermutation_create = partial(
    G_channelPermutation_create, mode="circular"
)
G_channelCircularPermutation_create.__name__ = "G_channelCircularPermutation_create"


def G_verticalFlip_create(Xshape):
    # Randomly flip an image vertically
    group = list(range(2))

    def action(X, gIdx):
        if group[gIdx]:
            X = np.flip(X, axis=0)
        return X

    return len(group), action


def G_horizontalFlip_create(Xshape):
    # Randomly flip an image horizontally
    group = list(range(2))

    def action(X, gIdx):
        if group[gIdx]:
            X = np.flip(X, axis=1)

        return X

    return len(group), action


def G_flip_create(Xshape):
    # Randomly flip an image vertically, horizontally or both
    group = list(itertools.product([0, 1], [0, 1]))

    def action(X, gIdx):
        g = group[gIdx]
        if g[0] == 1:
            X = np.flip(X, axis=0)
        if g[1] == 1:
            X = np.flip(X, axis=1)
        return X

    return len(group), action


def randomGroupTransformation(imageData, createGroupAction, changeTarget=None):
    # Given the entire image data, apply random transformations to each image
    #   according to the given group.
    # Should change the label of the image if the group affects the label (given by `changeTarget`).
    data = imageData.data
    targets = imageData.targets
    Xs = data.numpy()

    m, groupAction = createGroupAction(Xs[0].shape)

    for i in range(Xs.shape[0]):
        if changeTarget is None:
            gIdx = np.random.choice(range(1, m))
        else:
            gIdx = np.random.choice(m)

        if changeTarget == "replace":
            # Ignore the original digit label. Simply use the transformation as a label.
            targets[i] = gIdx
        elif changeTarget == "add":
            # New label is both the original digit label and the exact transformation label.
            targets[i] = targets[i] * m + gIdx
        elif changeTarget == "modifyBal":
            # New label is a function of the original digit label and the transformation label.
            targets[i] = targets[i] * 2 + (gIdx >= m // 2)

        # Apply the transformation to the image
        Xs[i] = groupAction(Xs[i], gIdx)

    imageData.data = torch.tensor(Xs)


def getCreateGroupsFromNames(names):
    # Group strings to functions
    mappingDict = {
        "rotation": G_rotation_create,
        "horizontal_flip": G_horizontalFlip_create,
        "vertical_flip": G_verticalFlip_create,
        "flip": G_flip_create,
        "color_permutation": G_channelPermutation_create,
        "channel_permutation": G_channelPermutation_create,
        "channel_circular_permutation": G_channelCircularPermutation_create,
        "color_circular_permutation": G_channelCircularPermutation_create,
    }

    return [mappingDict[name.lower()] for name in names]


def createAllHypothesis(datasetName, hypList, groups):
    # Create different extrapolation datasets (with different subsets G_I and G_D)
    dataDir = "data"
    for hyp in hypList:
        mode = hyp
        fold = (0, 5)
        trainData = ExtrapolationDataset(
            dataDir,
            name=datasetName,
            groups=groups,
            mode=mode,
            train=True,
            fold=fold,
            pSamples=None,
            transform=transforms.ToTensor(),
            normalize_transform=True,
            download=True,
            force=False,
        )
        testData = ExtrapolationDataset(
            dataDir,
            name=datasetName,
            groups=groups,
            mode=mode,
            fold=fold,
            train=False,
            transform=transforms.ToTensor(),
            normalize_transform=trainData.normalize_transform,
        )

        showAll(trainData, 10, title="Training data")
        showAll(testData, 10, title="Test data")
        print(f"Labels: {trainData.targets.unique()}")
        print("=" * 80)

    return trainData, testData


if __name__ == "__main__":
    dataset = "mnist"
    groups = "rotation_color_vflip"
    labelType = "add"

    # Construct the list of groups from a simple string parameter.
    # Additionally, forbid some subsets G_I and G_D to ensure the normal subgroup property (see Theorem 2 in the paper).
    if groups == "rotation":
        datasetGroups = ["rotation"]
    elif groups == "rotation_color":
        datasetGroups = ["rotation", "color_permutation"]
    elif groups == "rotation_color_flip":
        datasetGroups = ["rotation", "color_permutation", "flip"]
        forbiddenPairs = [(2, 0)]  # G_flip and G_rot have to be together.
    elif groups == "rotation_color_vflip":
        datasetGroups = ["rotation", "color_permutation", "vertical_flip"]
        forbiddenPairs = [(2, 0)]  # G_vflip and G_rot have to be together.
    elif groups == "rotation_color_hflip":
        datasetGroups = ["rotation", "color_permutation", "horizontal_flip"]
        forbiddenPairs = [(2, 0)]  # G_hflip and G_rot have to be together.
    else:
        raise NotImplementedError

    if dataset == "mnist" or dataset == "mnistfull":
        datasetGroups = [
            "channel_circular_permutation"
            if dg == "channel_permutation" or dg == "color_permutation"
            else dg
            for dg in datasetGroups
        ]

    datasetGroups = getCreateGroupsFromNames(datasetGroups)

    # All the different subsets possible (i.e., different values for G_I and G_D)
    hypList = [
        "".join(p) for p in itertools.product(["0", "1"], repeat=len(datasetGroups))
    ]

    # Remove cases where G_I is not a normal subgroup of G_{I \cup D}.
    newHypList = []
    for hyp in hypList:
        newHyp = list(hyp)
        flag = 0
        for fp in forbiddenPairs:
            if newHyp[fp[0]] == "0" and newHyp[fp[1]] == "1":
                flag = 1
            if newHyp[fp[0]] == "1" and newHyp[fp[1]] == "0":
                flag = 1
        if not flag:
            newHypList.append("".join(newHyp))

    hypList = list(itertools.product(["hyp1"], newHypList, [labelType]))
    hypList = ["_".join(hyp) for hyp in hypList]

    # Create all datasets and save in folder.
    createAllHypothesis(dataset, hypList[0:1], datasetGroups)
