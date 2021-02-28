import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Sequences.models.mlp import MLP
from Sequences.datasets import *
from tqdm.autonotebook import tqdm
import invariantSubspaces as IS
import itertools
from functools import partial
from Sequences.models.CGSequenceLayer import CGSequenceLayer
import signal
from Common import utils
import os

class CGFF(nn.Module, utils.GracefulKiller):
    """
    CG-Feedforward network for Sequences.
    The input is a sequence of length n and dimension d (X \in \mathbb{R}^{n\times d}).
    First, a point-wise feedforward network \phi is applied to each element in the sequence.
    Then we apply a CGSequenceLayer that takes as parameter a list of m groups (respective Reynolds operator functions)
    applied on sequences. For example, we will typically use \binom{n}{2} transposition groups. The layer learns to be
    invariant to a subset of the groups that do not affect the label.
    See Figure 7 in the paper for an example architecture.
    """

    def __init__(self, **modelParams):
        """
        :param modelParams: vocabularySize (int): Vocabulary size
                            embeddingSize (int): Embedding Layer size
                            phiLayerDims (list): [phi_hidden1, phi_hidden2, ... ]
                            rhoLayerDims (list): [rho_hidden1, rho_hidden2, ... outputSize]
                            dropout (float): Dropout probability
                            activation (string): Activation class like "relu"
                            outputActivation (string): Output activation class like "softmax"
        """

        super().__init__()

        self.sequenceLength = modelParams["sequenceLength"]
        vocabularySize = modelParams["vocabularySize"]
        embeddingSize = modelParams["embeddingSize"]
        phiLayerDims = modelParams["phiLayerDims"]
        rhoLayerDims = modelParams["rhoLayerDims"]
        hiddenDimension = modelParams["hiddenDimension"]

        dropout = modelParams.get("dropout", None)
        activation = modelParams.get("activation", "none")
        outputActivation = modelParams.get("outputActivation", "none")
        weightsAcrossDims = modelParams.get("weightsAcrossDims", False)

        # Folder name where the subspaces (the basis vectors) are stored.
        self.precomputedBasisFolder = modelParams.get("precomputedBasisFolder", ".")

        # Fixed embedding layer for sequence elements
        self.embeddingLayer = nn.Embedding(vocabularySize, embeddingSize)
        self.embeddingLayer.weight.requires_grad = False

        # Strength and temperature for the penalty.
        self.penaltyAlpha = modelParams.get("penaltyAlpha", 0)
        self.penaltyMode = modelParams.get("penaltyMode", "simple")
        self.penaltyT = modelParams.get("penaltyT", 1)

        os.makedirs(self.precomputedBasisFolder, exist_ok=True)

        # Reynolds operators for the transposition groups
        listT = []
        for ij in list(itertools.combinations(range(self.sequenceLength), 2)):
            f = partial(IS.G_permutation, pos=ij)
            f.__name__ = f"({ij[0]} {ij[1]})"
            listT.append(f)

        # Elementwise feedforward network
        self.phi = MLP(
            layerDims=[embeddingSize] + phiLayerDims,
            dropout=dropout,
            activation=activation,
            intermediate=True,
            outputActivation="none",
        )

        # CG-invariant sequence layer
        self.cgLayer = CGSequenceLayer(input_size=self.sequenceLength,
                                       input_dimension=phiLayerDims[-1],
                                       hidden_dimension=hiddenDimension,
                                       invariant_transforms=listT,
                                       precomputed_basis_folder=self.precomputedBasisFolder,
                                       weightsAcrossDims=weightsAcrossDims,
                                       bias=True)

        # Dense layer after the CG-invariant sequence layer
        self.rho = MLP(
            layerDims=[hiddenDimension] + rhoLayerDims,
            dropout=dropout,
            activation=activation,
            outputActivation=outputActivation,
        )

        # Signals to be handled by the GracefulKiller parent class.
        # At these signals, sets self.kill_now to True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)


    def forward(self, X):
        out = self.embeddingLayer(X)
        out = self.phi(out)
        out = self.cgLayer(out)
        out = self.rho(out)
        return out.squeeze()

    def computeLoss(self, data, loader=None):
        if data:
            _, X, y = data[:]

            device = utils.getDevice()
            X, y = X.to(device), y.to(device)

            outs = self.forward(X)
            loss = ((outs - y) ** 2).mean()

            penalty = torch.tensor(0)
            if self.penaltyAlpha > 0:
                penalty = self.penaltyAlpha * self.cgLayer.penalty(mode=self.penaltyMode, T=self.penaltyT)
        elif loader:
            # Loader for the full dataset is given
            # Returns loss in float (no grad).
            loss = 0
            device = utils.getDevice()
            with torch.no_grad():
                for batchIdx, batchData in enumerate(loader):
                    _, X, y = batchData[:]
                    X, y = X.to(device), y.to(device)

                    outs = self.forward(X)

                    loss += ((outs - y)**2).sum()

                loss /= len(loader.dataset)

            penalty = torch.tensor(0)
            if self.penaltyAlpha > 0:
                penalty = self.penaltyAlpha * self.cgLayer.penalty(mode=self.penaltyMode, T=self.penaltyT)
        else:
            raise AttributeError

        return loss + penalty, {"loss": loss, "penalty": penalty}

    def fit(self, trainData, validData, fitParams):
        """
        Fit the model to the training data
        Parameters
        ----------
        trainData : Train Dataset
        validData : Validation Dataset
        fitParams : Dictionary with parameters for fitting.
                    (lr, weightDecay(l2), lrSchedulerStepSize, fileName, batchSize, lossName, patience, numEpochs)
        Returns
        -------
        None
        """
        log = utils.getLogger()
        tqdmDisable = fitParams.get("tqdmDisable", False)
        optimizer = optim.Adam(
            self.parameters(), lr=fitParams["lr"], weight_decay=fitParams["weightDecay"]
        )
        fileName = fitParams["fileName"]

        N = trainData.shape[0]
        if fitParams.get("nMinibatches"):
            batchSize = 1 << int(math.log2(N // fitParams["nMinibatches"]))
        else:
            batchSize = fitParams.get("batchSize", 0)

        if batchSize == 0 or batchSize >= len(trainData):
            batchSize = len(trainData)

        bestValidLoss = np.inf
        patience = fitParams["patience"]
        numEpochs = fitParams["numEpochs"]
        validationFrequency = 1

        trainLoader = DataLoader(trainData, batch_size=batchSize)

        trainLoss, trainDict = self.computeLoss(data=None, loader=trainLoader)
        trainLoss = trainLoss.item()
        trainPenalty = trainDict["penalty"].item()

        for epoch in tqdm(range(1, numEpochs + 1), leave=False, desc="Epochs", disable=tqdmDisable):
            validLoss, validDict = self.computeLoss(validData)
            validLoss = validLoss.item()

            saved = ""
            if (
                    epoch == 1
                    or epoch > patience
                    or epoch >= numEpochs
                    or epoch % validationFrequency == 0
            ) and validLoss < bestValidLoss:
                saved = "(Saved model)"
                torch.save(self, fileName)
                if validLoss < 0.995 * bestValidLoss :
                    patience = np.max([epoch * 2, patience])
                bestValidLoss = validLoss

            if epoch > patience:
                break

            log.info(
                f"{epoch} out of {min(patience, numEpochs)} | "
                f"Train Loss: {trainLoss:.4f} (Penalty: {trainPenalty:.4f}) | "
                f"Valid Loss: {validLoss:.4f} (Penalty: {validDict['penalty'].item():.4f}) | "
                f"{saved}\r"
            )

            trainLoss = 0.0
            trainPenalty = 0.0
            batchIdx = 0
            for batchIdx, batchTrainData in enumerate(
                    tqdm(trainLoader, leave=False, desc="Minibatches", disable=tqdmDisable)
            ):
                optimizer.zero_grad()  # zero the gradient buffer
                batchTrainLoss, batchTrainDict = self.computeLoss(batchTrainData)
                trainLoss += batchTrainLoss.item()
                trainPenalty += batchTrainDict["penalty"].item()
                batchTrainLoss.backward()
                optimizer.step()

            trainLoss /= batchIdx + 1
            trainPenalty /= batchIdx + 1

            if self.kill_now:
                break

    def test(self, testData):
        """
        Test the model
        """

        device = utils.getDevice()

        testLoader = DataLoader(
            testData, batch_size=128, num_workers=2, pin_memory=True
        )

        testLoss = 0
        testAcc = 0

        results = {}
        with torch.no_grad():
            for batchIdx, batchData in enumerate(testLoader):
                _, X, y = batchData[:]
                X, y = X.to(device), y.to(device)

                outs = self.forward(X)
                testLoss += ((outs - y)**2).sum().item()
                testAcc += (y == torch.round(outs)).float().sum().item()

        testLoss /= len(testData)
        testAcc = testAcc * 100 / len(testData)

        results["loss"] = testLoss
        results["accuracy"] = testAcc

        penalty = torch.tensor(0.0)
        if self.penaltyAlpha > 0:
            penalty = self.penaltyAlpha * self.cgLayer.penalty(mode=self.penaltyMode, T=self.penaltyT)
        results["penalty"] = penalty.item()
        results["lossAndPenalty"] = results["loss"] + results["penalty"]

        return results


def run(trainData, testData, _config):
    """
    Run model on given train, test data and compute metrics
    """
    device = utils.getDevice()

    # Use 20% of the training data as validation (for early stopping, hyperparameter tuning)
    trainIdx, validIdx = utils.getRandomSplit(trainData.shape[0], [80, 20])

    validData = SubsetStar(trainData, validIdx)
    trainData = SubsetStar(trainData, trainIdx)

    model = CGFF(**_config).to(device)

    model.train()
    model.fit(
        trainData=trainData,
        validData=validData,
        fitParams=_config,
    )

    model = torch.load(_config["fileName"]).to(device)
    model.eval()

    trainResults = model.test(trainData)
    validResults = model.test(validData)

    testResults = {}
    for key, testData_ in testData.items():
        testResults_ = model.test(testData_)
        testResults[key] = testResults_

    return trainResults, validResults, testResults


if __name__ == '__main__':
    basisDir = "data/basis"
    os.makedirs(basisDir, exist_ok=True)

    device = utils.getDevice()
    vocabularySize = 100
    nSamples = 10000

    task = SumTaskDataset
    print(task)

    trainData = task(
        nSamples=nSamples, sequenceLength=10, vocabularyRange=(1, vocabularySize), inputTransform=np.sort,
    )
    testDataOOD = task(
        nSamples=nSamples, sequenceLength=10, vocabularyRange=(1, vocabularySize)
    )

    testData = {"ood": testDataOOD}

    params = {
        # Model Params
        "sequenceLength": 10,
        "vocabularySize": vocabularySize,
        "embeddingSize": 128,
        "hiddenDimension": 128,
        "phiLayerDims": [128],
        "rhoLayerDims": [128, 1],
        "activation": "relu",
        "precomputedBasisFolder": basisDir,
        "weightsAcrossDims": False,
        "penaltyMode": "simple",
        "penaltyT": 1,
        "penaltyAlpha": 10,

        # Fit Params
        "lr": 1e-3,
        "weightDecay": 0,
        "batchSize": 128,
        "numEpochs": 100,
        "patience": 100,
        "fileName": "model.pkl",
    }


    trainResults, validResults, testResults = run(
        trainData,
        testData,
        _config=params,
    )

    print(trainResults)
    print(testResults)


