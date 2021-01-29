import argparse
import random
import logging
import os
import sys
sys.path.append(os.path.abspath("./"))
from Common import utils

###########################
from Sequences.models import *
from Sequences.datasets import *
###########################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extrapolation via G-invariance on Sequences')

    parser.add_argument('--dataDir', type=str, default='data', metavar='dataDir', help='Data directory (to save basis) [default: data]')
    parser.add_argument('--dataset', type=str, metavar='dataset', help='One of [Sum|Sum2|EvenMinusOddSum|GeometricDistribution]TaskDataset')
    parser.add_argument('--nSamples', type=int, default=10000, metavar='nSamples', help='Number of samples [default: 10000]')
    parser.add_argument('--model', type=str, metavar='model', help='Model: cgff')
    parser.add_argument('--weightsAcrossDims', action='store_true', default=False, help='Different weights across different dimensions of the input.')
    parser.add_argument('--penaltyAlpha', type=float, default=10, metavar='alpha', help='Penalty strength [default: 10]')
    parser.add_argument('--penaltyMode', type=str, default='simple', metavar='penaltyMode', help='L0 approximation (simple or sigmoid) [default: simple]')
    parser.add_argument('--penaltyT', type=float, default=1, metavar='temperature', help='L0 approximation temperature [default: 1]')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout', help='Dropout [default: 0.0]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='lr', help='Learning rate [default: 1e-2]')
    parser.add_argument('--weightDecay', type=float, default=0, metavar='weightDecay', help='Weight decay [default: 0]')
    parser.add_argument('--batchSize', type=int, default=64, metavar='batchSize', help='Batch Size [default: 64]')
    parser.add_argument('--numEpochs', type=int, default=1000, metavar='numEpochs', help='Epochs [default: 1000]')
    parser.add_argument('--logLevel', type=str, default="info", metavar='logLevel', help='Log level [default: info]')
    parser.add_argument('--seed', type=int, default=314271, metavar='seed', help='Seed [default: 314271 (PIE)]')
    parser.add_argument('--tqdmDisable', action='store_true', default=False, help='Disable tqdm progress bars')

    args = parser.parse_args()
    _config = vars(args)

    # Additional architecture params
    _config["embeddingSize"] = 128
    _config["sequenceLength"] = 10
    _config["vocabularyRange"] = [1, 100]
    _config["vocabularySize"] = _config["vocabularyRange"][1]
    _config["activation"] = "relu"
    _config["hiddenDimension"] = 128
    _config["phiLayerDims"] = [128]
    _config["rhoLayerDims"] = [128, 1]
    _config["patience"] = max(10, _config["numEpochs"] // 10)

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

    device = utils.getDevice()
    _log.info(f"Using {device}")

    # Validate the current configuration
    utils.logInfoDict(_log, _config, "Configuration: ")

    datasetClass = utils.extractFromGlobals(globals(), _config["dataset"], SequenceDataset)

    # Sample train/test datasets.
    nSamples = _config["nSamples"]
    sequenceLength = _config["sequenceLength"]
    vocabularyRange = _config["vocabularyRange"]
    trainData = datasetClass(
        nSamples * 8 // 10,
        sequenceLength,
        vocabularyRange,
        inputTransform=np.sort
    )
    testData = {}
    testData["in"] = datasetClass(nSamples * 2 // 10, sequenceLength, vocabularyRange, inputTransform=np.sort)
    testData["ood"] = datasetClass(nSamples * 2 // 10, sequenceLength, vocabularyRange)

    _log.info(f"Train Dataset: {datasetClass.__name__} of shape {trainData.shape}")
    for testData_ in testData.values():
        _log.info(f"Test Dataset: {datasetClass.__name__} of shape {testData_.shape}")

    # Add temporary files in the configuration dictionary
    tempFileName, = utils.createTempFiles(
        folder=os.path.join(_config["dataDir"], "_tmp"), suffixList=[""]
    )

    # Add filenames to save models
    _config["fileName"] = tempFileName

    precomputedBasisFolder = os.path.join(_config["dataDir"], "basis")
    _config["precomputedBasisFolder"] = precomputedBasisFolder

    # Run the model
    model = _config["model"]
    module = utils.extractFromGlobals(globals(), model, None)
    _log.info(f"Begin Run : {model} => {module.__name__}")
    runFunction = module.run
    trainResults, validResults, testResults = runFunction(
        trainData,
        testData,
        _config=_config,
    )

    utils.logInfoDict(_log, trainResults, "Train Results:")
    utils.logInfoDict(_log, validResults, "Validation Results:")
    utils.logInfoDict(_log, testResults, "Test Results:")

