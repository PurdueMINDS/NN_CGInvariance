import argparse
import sys
import os
sys.path.append(os.path.abspath("./"))

import random
import numpy as np
import torch
from Common import utils
from torchvision import transforms
import logging

###########################
from Images.models import *
import Images.datasets as datasets
from Images.metrics import *
###########################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extrapolation via G-invariance on Images')

    parser.add_argument('--dataDir', type=str, default='data', metavar='dataDir', help='Data directory [default: data]')
    parser.add_argument('--dataset', type=str, metavar='dataset', help='mnistxtra (for MNIST34) or mnistfullxtra (for MNISTFull)')
    parser.add_argument('--groups', type=str, metavar='groups', help='One of [rotation, rotation_color, rotation_color_vflip, rotation_color_hflip, rotation_color_flip]')
    parser.add_argument('--datasetMode', type=str, default='000', metavar='datasetMode', help='Which groups in G_I (0) and which in G_D (1) [default: 000 (all of them in G_I)]')
    parser.add_argument('--model', type=str, metavar='model', help='Model: cgcnn')
    parser.add_argument('--architecture', type=str, default='vgg', metavar='architecture', help='simple (LeNet) or vgg architecture [default: vgg]')
    parser.add_argument('--penaltyAlpha', type=float, default=10, metavar='alpha', help='Penalty strength [default: 10]')
    parser.add_argument('--penaltyMode', type=str, default='simple', metavar='penaltyMode', help='L0 approximation (simple or sigmoid) [default: simple]')
    parser.add_argument('--penaltyT', type=float, default=1, metavar='temperature', help='L0 approximation temperature [default: 1]')
    parser.add_argument('--cvIt', type=int, default=0, metavar='cvIt', help='i-th iteration of cross-validation [default: 0]')
    parser.add_argument('--cvFolds', type=int, default=5, metavar='cvFolds', help='k-fold cross-validation [default: 5]. Set -1 for no cross-validation.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum', help='Momentum for SGD [default: 0.9]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='lr', help='Learning rate [default: 1e-3]')
    parser.add_argument('--weightDecay', type=float, default=0, metavar='weightDecay', help='Weight decay [default: 0]')
    parser.add_argument('--batchSize', type=int, default=64, metavar='batchSize', help='Batch Size [default: 64]')
    parser.add_argument('--numEpochs', type=int, default=500, metavar='numEpochs', help='Epochs [default: 500]')
    parser.add_argument('--logLevel', type=str, default="info", metavar='logLevel', help='Log level [default: info]')
    parser.add_argument('--seed', type=int, default=314271, metavar='seed', help='Seed [default: 314271 (PIE)]')
    parser.add_argument('--tqdmDisable', action='store_true', default=False, help='Disable tqdm progress bars')

    args = parser.parse_args()
    _config = vars(args)
    _config["patience"] = max(10, _config["numEpochs"] // 10)       # Patience for early-stopping

    # Setup logger for the experiment
    # Remove the root logger
    # Add Tqdm logger setup to print along with the progress bar.
    rootLogger = logging.getLogger()
    rootLogger.handlers = []
    # Whenever a logger is required, query using logging.getLogger("_log").
    _log = logging.getLogger("_log")
    _log.handlers = []
    tqdmHandler = utils.TqdmLoggingHandler()  # Works with tqdm progress bars.
    tqdmHandler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
    _log.addHandler(tqdmHandler)
    _log.setLevel(_config["logLevel"].upper())

    # Set seed for random, numpy and torch (for reproducibility)
    seed = _config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create data directory
    os.makedirs(_config['dataDir'], exist_ok=True)

    device = utils.getDevice()
    _log.info(f"Using {device}")

    # Print the current configuration
    utils.logInfoDict(_log, _config, "Configuration: ")

    # If performing cross-validation, obtain the fold number.
    if _config["cvIt"] == -1 or _config["cvFolds"] == -1:
        fold = None
    else:
        fold = (_config["cvIt"], _config["cvFolds"])


    # Parse the --groups command to obtain the individual `m` number of groups.
    # List of groups for each layer of the CNN. This is same for all layers except the first layer
    #   which may have the color-permutation group. Assumes a maximum of 8 CGConv2D layers.
    groups = _config["groups"]
    if groups == "rotation":
        datasetGroups = ["rotation"]
        listInvariantTransforms = [["rotation"]] * 8
    elif groups == "rotation_color":
        datasetGroups = ["rotation", "color_permutation"]
        listInvariantTransforms = [["rotation", "color_permutation"]] + [["rotation"]] * 7
    elif groups == "rotation_color_flip":
        datasetGroups = ["rotation", "color_permutation", "flip"]
        listInvariantTransforms = [["rotation", "color_permutation", "flip"]] + [["rotation", "flip"]] * 7
    elif groups == "rotation_color_vflip":
        datasetGroups = ["rotation", "color_permutation", "vertical_flip"]
        listInvariantTransforms = [["rotation", "color_permutation", "vertical_flip"]] + [["rotation", "vertical_flip"]] * 7
    elif groups == "rotation_color_hflip":
        datasetGroups = ["rotation", "color_permutation", "horizontal_flip"]
        listInvariantTransforms = [["rotation", "color_permutation", "horizontal_flip"]] + [["rotation", "horizontal_flip"]] * 7
    else:
        raise NotImplementedError
    _config["listInvariantTransforms"] = listInvariantTransforms

    dataset = _config["dataset"]
    if dataset.lower() == "mnistxtra" or dataset.lower() == "mnistfullxtra":
        datasetGroups = ["channel_circular_permutation" if dg=="channel_permutation" or dg=="color_permutation" else dg for dg in datasetGroups]
        createGroups = datasets.getCreateGroupsFromNames(datasetGroups)
        datasetMode = _config["datasetMode"]
        datasetMode = 'hyp1_' + datasetMode + ('_add' if dataset.lower()=="mnistxtra" else '_modifyBal')

        # Create training data by randomly transforming the MNIST training data by the groups in G_D
        # (this is given by the --datasetMode parameter).
        trainData = datasets.ExtrapolationDataset(
            root=_config["dataDir"],
            name=dataset.lower()[:-4],
            groups=createGroups,
            mode=datasetMode,
            train=True,
            fold=fold,
            pSamples=None,
            transform=transforms.ToTensor(),
            normalize_transform=False,
            download=True,
        )

        # Create extrapolated test data by randomly transforming the MNIST test data by the groups in G_I
        # (this is given by the --datasetMode parameter).
        testData = {}
        testData["ood"] = datasets.ExtrapolationDataset(
            root=_config["dataDir"],
            name=dataset.lower()[:-4],
            groups=createGroups,
            mode=datasetMode,
            fold=fold,
            train=False,
            transform=transforms.ToTensor(),
            normalize_transform=False,
        )
        nOutputs = len(trainData.targets.unique())
        _log.info(f"Number of classes: {nOutputs}")

        _config["imageSize"] = 28
        _config["inputChannels"] = 3
        _config["nOutputs"] = nOutputs


    _log.info(f"Train Dataset: {dataset} of shape {len(trainData)}")
    for key, testData_ in testData.items():
        _log.info(f"Test Dataset ({key}): {dataset} of shape {len(testData_)}")

    # Add temporary files in the configuration dictionary
    tempFileName, = utils.createTempFiles(
        folder=os.path.join(_config["dataDir"], "_tmp"), suffixList=[""]
    )

    # Add filenames to save models
    _config["fileName"] = tempFileName

    # The basis for the 1-eigenspaces will be stored here after the first computation.
    precomputedBasisFolder = os.path.join(_config["dataDir"], "basis")
    _config["precomputedBasisFolder"] = precomputedBasisFolder

    # Obtain metrics
    metrics = [accuracy_noavg]

    # Run the model
    model = _config["model"]
    module = utils.extractFromGlobals(globals(), model, None)
    _log.info(f"Begin Run : {model} => {module.__name__}")
    runFunction = module.run
    trainResults, validResults, testResults = runFunction(
        trainData,
        testData,
        metrics=metrics,
        _config=_config,
    )

    utils.logInfoDict(_log, trainResults, "Train Results:")
    utils.logInfoDict(_log, validResults, "Validation Results:")
    utils.logInfoDict(_log, testResults, "Test Results:")

