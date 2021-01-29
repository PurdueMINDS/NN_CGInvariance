import numpy as np
import torch
import os
import logging
from tqdm.auto import tqdm
from pprint import pformat
import tempfile
from fuzzywuzzy import process
import inspect
import random


################################ Torch helpers ########################################
device = None
def getDevice():
    global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def getLogger():
    log = logging.getLogger("_log")

    if log.handlers == []:
        rootLogger = logging.getLogger()
        rootLogger.handlers = []
        tqdmHandler = TqdmLoggingHandler()  # Works with tqdm progress bars.
        tqdmHandler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
        log.addHandler(tqdmHandler)
        log.setLevel("INFO")

    return log

class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler to allow printing in between steps of tqdm bar.
    """
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class GracefulKiller:
    """
    Handles signals (like KeyboardInterrupt) gracefully
    Usage: Inherit from this class, add signal.signal() function with self.exit_gracefully as the handler.
           Sets self.kill_now to True if signal received.
    """
    kill_now = False
    def __init__(self):
        pass

    def exit_gracefully(self, signum, frame):
        logger = logging.getLogger("_log")
        logger.setLevel(logging.DEBUG)
        if not self.kill_now:
            logger.info("Stop signal received. Killing now...")
        else:
            logger.info("Processing. Please wait.")
        self.kill_now = True


def logInfoDict(logger, dict, text):
    """
    Logs (at level INFO) a dictionary in a pretty way (over different lines while logging).
    :param logger: Logger object
    :param dict: Dictionary to print
    :param text: Title
    :return:
    """
    logger.info("=" * 80)
    logger.info(text)
    for line in pformat(dict).split("\n"):
        logger.info(line)
    logger.info("=" * 80)


def createTempFiles(folder, suffixList=[]):
    """
    Create temporary files for model and plot
    :param folder: Folder to create temporary files in
    :param suffixList: List of suffixes. Model files have no suffix, plot files need to have '.pdf' suffix.

    """

    os.makedirs(folder, exist_ok=True)
    tempFiles = []

    for suffix in suffixList:
        _, tempFileName = tempfile.mkstemp(suffix=suffix, dir=folder)
        tempFiles.append(tempFileName)

    return tempFiles


def getRandomSplit(N, splitPercentages):
    """
    Get random indices to split the data according to the percentages specified
    :param N: Generate a permutation from (0, N-1)
    :param splitPercentages: Percentages to split
    :return: List of indices with lengths according to splitPercentages
    """
    assert np.sum(splitPercentages) == 100, 'Percentages should sum to 100'

    indices = np.random.permutation(N)
    splitValues = (np.cumsum(splitPercentages[:-1]) * N / 100).astype(int)
    return np.split(indices, splitValues)


def getActivationClassFromString(activationClassString):
    """

    Parameters
    ----------
    activationClassString: String like "ReLU, Tanh"

    Returns
    -------
    Activation class

    """
    import torch.nn as nn
    if activationClassString is None:
        return None

    activationDictionary = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "none": None
    }

    return activationDictionary[activationClassString.lower()]


def extractFromGlobals(modules, string, parentClass):
    """
    Extract the class from a string
    :param string: Name of the class
    :param parentClass: Parent class (for matching)
    :return:
    """

    if parentClass:
        # Add all child classes to search list.
        keys = [name for name, cls in modules.items() if inspect.isclass(cls) and issubclass(cls, parentClass)]
    else:
        # No parent class specified.
        # Add all modules to search list (no classes allowed).
        keys = [name for name, mods in modules.items() if not inspect.isclass(mods)]

    className, score = process.extractOne(string, keys)
    if score <= 50:
        raise AttributeError(f"Could not find a good match for '{string}' of class {parentClass}.")

    return modules[className]

