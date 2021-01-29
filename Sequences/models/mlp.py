import torch
import torch.nn as nn
from Common import utils


class MLP(nn.Module):
    """
    Simple feedforward network
    """

    def __init__(self, **modelParams):
        """
        Initialize an MLP
        :param modelParams: layerDims (list): [inputSize, hidden1, hidden2, ..., outputSize]
                            dropout (float): Dropout probability
                            batchNorm (bool): Use batch normalization or not
                            activation (string): Activation class like "relu"
                            intermediate (bool): True if the MLP is intermediate and does not contain the final layer (default: False)
                            outputActivation (string): Output activation class like "softmax"
        """

        super().__init__()

        layerDims = modelParams["layerDims"]
        dropout = modelParams.get("dropout", None)
        batchNorm = modelParams.get("batchNorm", None)
        activationClass = utils.getActivationClassFromString(modelParams.get("activation", "none"))
        intermediate = modelParams.get("intermediate", False)
        outputActivationClass = utils.getActivationClassFromString(
            modelParams.get("outputActivation", "none")
        )

        self.net = nn.Sequential()

        for i in range(len(layerDims) - 1):
            inputDim = layerDims[i]
            outputDim = layerDims[i + 1]
            self.net.add_module(
                f"dense_{i+1}", nn.Linear(inputDim, outputDim, bias=True)
            )

            # No activation, Batch normalization, Dropout for last layer.
            # LayerDims: [input, h1, h2, ... hn, output]. When i=len(layerDims)-2, then we are at the last layer.
            if intermediate or (i < len(layerDims) - 2):
                if dropout:
                    self.net.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))
                if batchNorm:
                    self.net.add_module(f"batchnorm_{i+1}", nn.BatchNorm1d(outputDim))
                if activationClass is not None:
                    self.net.add_module(f"activation_{i+1}", activationClass())


        if outputActivationClass is not None:
            self.net.add_module(f"outputActivation", outputActivationClass(dim=1))

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    # Usage

    inputSize = 10
    nSamples = 1000
    X = torch.rand([nSamples, inputSize])

    modelParams = {
        "layerDims": [inputSize, 128, 2],
        "dropout": None,
        "batchNorm": None,
        "activation": "relu",
        "outputActivation": "softmax",
    }

    # net = MLP(**modelParams)
    net = MLP(layerDims=[inputSize, 128, 2], outputActivation="softmax")
    output = net(X)

    print(output)


