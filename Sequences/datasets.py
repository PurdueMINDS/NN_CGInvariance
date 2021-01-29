import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC
import matplotlib.pyplot as plt


class SequenceDataset(ABC, Dataset):
    """
    Abstract class for Sequence datasets
    Data consists of sequences (arranged in rows).
    X : sequence
    y : label
    """

    @property
    def shape(self):
        """
        Shape
        """
        return self.X.shape

    @property
    def xshape(self):
        """
        Shape of covariates X
        """
        return self.X.shape

    @property
    def yshape(self):
        """
        Shape of labels y
        """
        return self.y.shape

    def __init__(self, X, y):
        if type(X) == torch.Tensor:
            self.X = X
            self.y = y.squeeze()
        else:
            self.X = torch.tensor(X)
            self.y = torch.tensor(y).squeeze()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Gets X and y from the dataframe. Note that the X and y are transposed for (batch, features) format.
        """
        X = self.X[idx]
        y = self.y[idx]

        return idx, X, y


class SubsetStar(Dataset):
    """
    Subset of a dataset at specified indices.
    Extended version of what is implemented in Pytorch.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getattr__(self, property):
        # Will be called if the property was not found in the current class
        # Check the original dataset for the specified property
        if hasattr(self.dataset, property):
            attr = getattr(self.dataset, property)
            if "shape" in property.lower():
                attr = len(self), *attr[1:]

            elif "dataframe" in property.lower():
                attr = attr.iloc[self.indices, :]
            return attr
        else:
            raise AttributeError

    def __getitem__(self, idx):
        id, X, y = self.dataset[self.indices[idx]]
        return id, X, y

    def __len__(self):
        return len(self.indices)


##################################################
#  Invariant/Partially-invariant/Sequence Tasks  #
##################################################
class SumTaskDataset(SequenceDataset):
    """
    Given a sequence $X = (x_1, ... x_n)$,
    predict $y = \sum_{i=1}^n x_i$,
    the sum of all the elements.
    G_I = S_{1...n}, G_D = {Id}
    """

    def __init__(self, nSamples, sequenceLength=10, vocabularyRange=(0, 100), inputTransform=None):
        """

        :param nSamples:  Number of samples
        :param sequenceLength:  Length of sequence
        :param vocabularyRange: Range of the vocabulary (a, b)
        :param inputTransform: A function applied to each input sequence. Used to add bias to the dataset.
        """

        outputFunc = np.sum

        input = np.random.choice(
            range(*vocabularyRange), (nSamples, sequenceLength), replace=True
        )

        if inputTransform:
            input = np.apply_along_axis(inputTransform, 1, input)

        output = np.apply_along_axis(outputFunc, 1, input).reshape(-1, 1)

        super().__init__(input, output)

    def _group(self):
        return [np.s_[:]]

class EvenMinusOddSumTaskDataset(SequenceDataset):
    """
    Given a sequence $X = (x_1, ... x_n)$,
    predict $y = \sum_{i=1}^{n/2} (x_{2i} - x_{2i+1})
    G_I = S_{{1,3,...}, {2,4,..}}, G_D = {Id}
    """

    def __init__(self, nSamples, sequenceLength, vocabularyRange, inputTransform=None):
        """

        :param nSamples:  Number of samples
        :param sequenceLength:  Length of sequence
        :param vocabularyRange: Range of the vocabulary (a, b)
        """

        outputFunc = lambda l: np.sum(l[0::2]) - np.sum(l[1::2])

        input = np.random.choice(
            range(vocabularyRange[0], vocabularyRange[1]//2), (nSamples//2, sequenceLength), replace=True
        )
        input2 = np.random.choice(
            range(vocabularyRange[1]//2, vocabularyRange[1]), (nSamples//2, sequenceLength), replace=True
        )
        if inputTransform:
            input = np.apply_along_axis(inputTransform, 1, input)
            input2 = np.apply_along_axis(inputTransform, 1, input2)[:, ::-1].copy()
        input = np.concatenate([input, input2], axis=0)

        output = np.apply_along_axis(outputFunc, 1, input).reshape(-1, 1)

        super().__init__(input, output)

    def _group(self):
        return [np.s_[0::2], np.s_[1::2]]


class Sum2TaskDataset(SequenceDataset):
    """
    Given a sequence $X = (x_1, ... x_n)$,
    predict $y = \sum_{i=2}^n x_i$.
    The task is not permutation invariant. The first element $x_1$ should not be included.
    G_I = S_{2...n}, G_D = {Id}
    """

    def __init__(self, nSamples, sequenceLength, vocabularyRange, inputTransform=None):
        """

        :param nSamples:  Number of samples
        :param sequenceLength:  Length of sequence
        :param vocabularyRange: Range of the vocabulary (a, b)
        """

        outputFunc = lambda l: np.sum(l[1:])

        input = np.random.choice(
            range(vocabularyRange[0], vocabularyRange[1]//2), (nSamples//2, sequenceLength), replace=True
        )
        input2 = np.random.choice(
            range(vocabularyRange[1]//2, vocabularyRange[1]), (nSamples//2, sequenceLength), replace=True
        )
        if inputTransform:
            input = np.apply_along_axis(inputTransform, 1, input)
            input2 = np.apply_along_axis(inputTransform, 1, input2)[:, ::-1].copy()

        input = np.concatenate([input, input2], axis=0)

        output = np.apply_along_axis(outputFunc, 1, input).reshape(-1, 1)

        super().__init__(input, output)

    def _group(self):
        return [np.s_[1:]]


class GeometricDistributionTaskDataset(SequenceDataset):
    """
    Given a sequence $X = (x_1, ... x_n)$,
    predict number of failures before first success, where success is x_i > threshold.
    G_I = {Id}, G_D = S_{1...n}
    """

    def __init__(self, nSamples, sequenceLength, vocabularyRange=None, inputTransform=None):
        """

        :param nSamples:  Number of samples
        :param sequenceLength:  Length of sequence
        :param vocabularyRange: Range of the vocabulary (a, b)
        """

        outputFunc = lambda l: (l >= mid).cumprod().sum()

        goodp = np.round(1 - np.exp(np.log(0.1)/sequenceLength), decimals=2)
        mid = goodp * (vocabularyRange[1] - vocabularyRange[0]) + vocabularyRange[0]

        input = np.random.choice(
            range(*vocabularyRange), (nSamples, sequenceLength), replace=True
        )

        output = np.apply_along_axis(outputFunc, 1, input).reshape(-1, 1)

        super().__init__(input, output)

    def _group(self):
        return []



if __name__ == "__main__":
    task = SumTaskDataset

    trainData = task(nSamples=1000, sequenceLength=10, vocabularyRange=(1, 10), inputTransform=np.sort)
    testData = task(nSamples=1000, sequenceLength=10, vocabularyRange=(1, 10))

    # _, X, y = trainData[:]
    # plt.hist(y)
    # plt.xlabel("Output of the task")
    # plt.show()

    print(task.__name__)
    print("Training example (sorted):")
    print(trainData[0][1], trainData[0][2])
    print("Test example (extrapolated):")
    print(testData[0][1], testData[0][2])

