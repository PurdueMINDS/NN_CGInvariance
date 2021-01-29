import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm
from Common import utils
import os
import signal
from Images.metrics import *
from Images.models.CGConv2D import CGConv2D
import invariantSubspaces as IS
import threading
import queue as thqueue
import time


class CGCNN(nn.Module, utils.GracefulKiller):
    def __init__(self, **modelParams):
        super().__init__()

        self.imageSize = modelParams.get("imageSize", 28)
        self.inputChannels = modelParams.get("inputChannels", 3)
        self.kernelSize = modelParams.get("kernelSize", 3)
        self.stride = modelParams.get("stride", 1)
        self.padding = modelParams.get("padding", 1)
        self.nOutputs = modelParams.get("nOutputs", 10)

        self.precomputedBasisFolder = modelParams.get("precomputedBasisFolder", ".")

        listInvariantTransforms = modelParams.get("listInvariantTransforms", [None] * 4)
        listInvariantTransforms = [
            ["trivial"] if it is None or it == [] else it
            for it in listInvariantTransforms
        ]

        _log = utils.getLogger()
        _log.info(f"Invariant transforms for layers: {listInvariantTransforms}")

        listInvariantTransforms = [
            IS.getTransformationsFromNames(it) for it in listInvariantTransforms
        ]
        self.listInvariantTransforms = listInvariantTransforms

        os.makedirs(self.precomputedBasisFolder, exist_ok=True)

        self.penaltyAlpha = modelParams.get("penaltyAlpha", 0)
        self.penaltyMode = modelParams.get("penaltyMode", "simple")
        self.penaltyT = modelParams.get("penaltyT", 1)

        architecture = modelParams.get("architecture", "simple")
        if architecture == "simple":
            architecture = [10, 10, 'M', 20, 20, 'M']
        elif self.imageSize >= 32:
            architecture = [64, 'M', 128, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']
        else:
            architecture = [64, 'M', 128, 'M', 128, 128, 'M', 128, 128, 'M']

        self.convLayers, linearSize = self._make_layers(architecture)
        self.fc1 = nn.Linear(linearSize, 50)
        self.fc2 = nn.Linear(50, self.nOutputs)

        # Signals to be handled by the GracefulKiller parent class.
        # At these signals, sets self.kill_now to True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)


    def _make_layers(self, layerDims):
        # Adapted from https://github.com/kuangliu/pytorch-cifar
        layers = []
        featureSize = self.imageSize
        currentInputChannels = self.inputChannels
        lastDim = -1
        layerK = 0

        for layerSize in layerDims:
            if layerSize == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                featureSize = (featureSize - 2) // 2 + 1
            else:
                layer = CGConv2D(
                    currentInputChannels,
                    layerSize,
                    invariant_transforms=self.listInvariantTransforms[layerK],
                    kernel_size=self.kernelSize,
                    stride=self.stride,
                    padding=self.padding,
                    precomputed_basis_folder=self.precomputedBasisFolder,
                    penaltyAlpha=self.penaltyAlpha,
                )

                layers += [layer,
                           nn.BatchNorm2d(layerSize),
                           nn.ReLU(inplace=True)]

                currentInputChannels = layerSize
                featureSize = (featureSize - self.kernelSize + 2 * self.padding) // self.stride + 1
                layerK += 1
                lastDim = layerSize

        return nn.Sequential(*layers), lastDim

    def forward(self, x):
        x = self.convLayers(x)
        x = x.sum(dim=[-2, -1], keepdim=False)  # torch.Size([64, 80])
        x = x.view(x.shape[0], -1)  # torch.Size([64, 80])
        x = F.relu(self.fc1(x))  # torch.Size([64, 50])
        x = F.dropout(x, training=self.training)  # torch.Size([64, 50])
        x = self.fc2(x)  # torch.Size([64, 10])
        return F.log_softmax(x, dim=1)

    def computePenalty(self):
        penalty = torch.tensor(0.0)
        if self.penaltyAlpha > 0:
            penalty = []
            for layer in self.convLayers:
                if layer.__class__.__name__ == 'CGConv2D':
                    penalty.append(layer.penalty(mode=self.penaltyMode, T=self.penaltyT))

            penalty = torch.stack(penalty).sum()
            penalty = self.penaltyAlpha * penalty
        return penalty

    def computeLoss(self, dataInGPU, loader=None):
        if dataInGPU:
            # Batch dataset is given
            X, y = dataInGPU

            outs = self(X)
            loss = F.nll_loss(outs, y)
            penalty = self.computePenalty()

        elif loader:
            # Loader for the full dataset is given
            # Returns loss in float (no grad).
            loss = 0
            device = utils.getDevice()
            with torch.no_grad():
                for batchIdx, batchData in enumerate(loader):
                    X, y = batchData[:]
                    X, y = X.to(device), y.to(device)
                    outs = self.forward(X)

                    loss += F.nll_loss(outs, y, reduction="sum").item()

                loss /= len(loader.dataset)

                penalty = self.computePenalty().item()
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
        device = utils.getDevice()
        optimizer = optim.SGD(
            self.parameters(), lr=fitParams["lr"], momentum=fitParams["momentum"]
        )

        fileName = fitParams["fileName"]

        N = len(trainData)
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

        trainLoader = DataLoader(
            trainData,
            batch_size=batchSize,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )
        validLoader = DataLoader(
            validData, batch_size=1000, shuffle=True, num_workers=1, pin_memory=True
        )

        trainLoss, _ = self.computeLoss(dataInGPU=None, loader=trainLoader)
        trainPenalty = _["penalty"]
        epochTime = 0

        for epoch in tqdm(range(1, numEpochs + 1), leave=False, desc="Epochs", disable=tqdmDisable):
            validLoss, _ = self.computeLoss(dataInGPU=None, loader=validLoader)
            validPenalty = _["penalty"]

            saved = ""
            if (
                epoch == 1
                or epoch > patience
                or epoch >= numEpochs
                or epoch % validationFrequency == 0
            ) and validLoss < bestValidLoss:
                # saved = "(Saved to {})".format(fileName)
                saved = "(Saved model)"
                torch.save(self, fileName)
                if validLoss < 0.995 * bestValidLoss:
                    patience = np.max([epoch * 2, patience])
                bestValidLoss = validLoss

            if epoch > patience:
                break

            log.info(
                f"{epoch} out of {min(patience, numEpochs)} | Train Loss: {trainLoss:.4f} ({trainPenalty:.4f}) | Valid Loss: {validLoss:.4f} ({validPenalty: .4f}) | {saved}"
            )

            trainLoss = 0.0
            trainPenalty = 0.0

            queue = thqueue.Queue(10)
            dataProducer = threading.Thread(target=producer, args=(device, queue, trainLoader))
            dataProducer.start()

            startTime = time.time()
            while True:
                batchIdx, X, y = queue.get()
                if X is None:
                    break
                optimizer.zero_grad()  # zero the gradient buffer
                batchTrainLoss, lossAndPenalty = self.computeLoss((X, y))
                batchTrainLoss.backward()
                optimizer.step()
                trainLoss += float(batchTrainLoss.detach().cpu())
                trainPenalty += float(lossAndPenalty["penalty"].detach().cpu())

            epochTime = time.time() - startTime
            log.debug(f"Time taken this epoch: {epochTime}")
            trainLoss /= batchIdx + 1
            trainPenalty /= batchIdx + 1

            if self.kill_now:
                break

    def test(self, testData, metrics=None):
        """
        Test the model and plot curves
        """

        testLoader = DataLoader(
            testData, batch_size=1000, num_workers=5, pin_memory=True
        )

        device = utils.getDevice()
        testLoss = 0

        queue = thqueue.Queue(10)
        dataProducer = threading.Thread(target=producer, args=(device, queue, testLoader))
        dataProducer.start()

        results = {}
        with torch.no_grad():
            while True:
                batchIdx, X, y = queue.get()
                if X is None:
                    break

                outs = self.forward(X)
                testLoss += F.nll_loss(outs, y, reduction="sum").item()

                if metrics:
                    for metric in metrics:
                        metricName = metric.__name__.split("_")[0]
                        if metricName in results:
                            results[metricName] += metric(y, outs, X=X, model=self)
                        else:
                            results[metricName] = metric(y, outs, X=X, model=self)

        testLoss /= len(testData)

        for metricName in results.keys():
            results[metricName] = results[metricName] / len(testData)

        results["loss"] = testLoss

        results["penalty"] = self.computePenalty().item()
        results["lossAndPenalty"] = results["loss"] + results["penalty"]

        return results


def producer(device, queue, loader):
    # Place our data on the Queue
    for batchIdx, batchTrainData in enumerate(loader):
        # print(f"Queued: {batchIdx}")
        batchTrainData[0], batchTrainData[1] = batchTrainData[0].to(device, non_blocking=True), batchTrainData[1].to(device, non_blocking=True)
        queue.put((batchIdx, batchTrainData[0], batchTrainData[1]))
    queue.put((batchIdx, None, None))


def run(trainData, testData, metrics, _config):
    """
    Run model on given train, test data and compute metrics
    """
    device = utils.getDevice()

    # Use 20% of the training data as validation (for early stopping, hyperparameter tuning)
    trainIdx, validIdx = utils.getRandomSplit(len(trainData), [80, 20])
    validData = Subset(trainData, validIdx)
    trainData = Subset(trainData, trainIdx)

    model = CGCNN(**_config).to(device)

    model.train()
    model.fit(
        trainData=trainData,
        validData=validData,
        fitParams=_config,
    )

    model = torch.load(_config["fileName"]).to(device)
    model.eval()

    trainResults = model.test(trainData, metrics)
    validResults = model.test(validData, metrics)

    testResults = {}
    for key, testData_ in testData.items():
        testResults_ = model.test(testData_, metrics)
        testResults[key] = testResults_

    return trainResults, validResults, testResults


if __name__ == "__main__":
    from Images.datasets import ExtrapolationDataset, getCreateGroupsFromNames, showAll
    from torchvision import transforms

    dataDir = "data"
    dataset = "MNISTXtra"

    groups = ["rotation", "vertical_flip", "color_circular_permutation"]
    inputChannels = 3
    imageSize = 28

    mode = "hyp1_000_111_add"
    print(f"Dataset: {mode}")

    createGroups = getCreateGroupsFromNames(groups)

    fold = None
    trainData = ExtrapolationDataset(
        dataDir,
        name="mnist",
        groups=createGroups,
        mode=mode,
        train=True,
        fold=fold,
        pSamples=None,
        transform=transforms.ToTensor(),
        normalize_transform=False,
        download=True,
    )

    testData = {}
    testData["ood"] = ExtrapolationDataset(
        dataDir,
        name="mnist",
        groups=createGroups,
        mode=mode,
        fold=fold,
        train=False,
        transform=transforms.ToTensor(),
        normalize_transform=False,
    )

    showAll(trainData, title='Training Data')
    showAll(testData["ood"], title='Test Data')
    nOutputs = len(trainData.targets.unique())
    print(f"Number of classes: {nOutputs}")


    params = {
        "imageSize": imageSize,
        "inputChannels": inputChannels,
        "nOutputs": nOutputs,
        "architecture": "vgg",
        "listInvariantTransforms": [["rotation", "vertical_flip", "color_permutation"]] + [["rotation", "vertical_flip"]] * 7,
        "precomputedBasisFolder": os.path.join(dataDir, "basis/"),
        "penaltyAlpha": 10,
        "penaltyMode": "simple",
        "penaltyT": 1,
        # FitParams
        "lr": 1e-3,
        "momentum": 0.9,
        "batchSize": 64,
        "weightDecay": 0,
        "fileName": "model.pkl",
        "patience": 100,
        "numEpochs": 1000,
    }


    trainResults, validResults, testResults = run(
        trainData,
        testData,
        metrics=[accuracy_noavg, cm_noavg],
        _config=params,
    )

    print(trainResults)
    print(testResults)
