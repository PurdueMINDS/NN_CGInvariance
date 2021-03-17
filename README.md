# Neural Networks for Learning Counterfactual G-Invariances from Single Environments

This repository is the official implementation of [Neural Networks for Learning Counterfactual G-Invariances from Single Environments](https://openreview.net/forum?id=7t1FcJUWhi3). 

In this work, we introduce a novel learning framework for single-environment extrapolations, where invariance to transformation groups is mandatory even without evidence, unless the learner deems it inconsistent with the training data. We also introduce sequence and image extrapolation tasks that validate our framework. 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Tutorial Notebook
For a quick tutorial on how the bases are obtained and used, check [cginvariance_example.ipynb](cginvariance_example.ipynb) (in case of rendering issues, download [cginvariance_example.html](cginvariance_example.html) and open in browser). 

The bases found by Theorem 3 in the paper for groups on Images and Sequences are shown in [invariantSubspaces.ipynb](invariantSubspaces.ipynb).

## Architectures & Experiments

### Images

To train the model in the paper, run this command:

```train
python Images/main.py --dataset=mnistxtra --groups=rotation_color_vflip --datasetMode='000' --model=cgcnn --numEpochs=500
```



Main arguments to **Images/main.py** are:

Dataset arguments

```
  --dataDir=dataDir         Data directory [default: data]
  --dataset=dataset         mnistxtra (for MNIST34) or mnistfullxtra (for MNISTFull)
  --groups=groups           One of [rotation, rotation_color, rotation_color_vflip, rotation_color_hflip, rotation_color_flip]
  --datasetMode=dm          Which groups in G_I (0) and which in G_D (1) [default: 000 (all of them in G_I)]
  --cvIt=cvIt               i-th iteration of cross-validation [default: 0]
  --cvFolds=cvFolds         k-fold cross-validation [default: 5]. Set -1 for no cross-validation.
```



Architecture/model arguments

```
  --model=model             Model: cgcnn
  --architecture=arch       simple (LeNet) or vgg architecture [default: vgg]
  --penaltyAlpha=alpha      Penalty strength [default: 10]
  --penaltyMode=mode        L0 approximation (simple or sigmoid) [default: simple]
  --penaltyT=T              L0 approximation temperature [default: 1]
```

Other arguments include ``--batchSize``, ``--numEpochs``, ``—lr``, ``—momentum ``, ``--seed`` with the usual meanings.



#### Image Example
An example training of CGCNN on MNIST-34 dataset (with groups=rotation_color_vflip) is shown in [image_example.ipynb](image_example.ipynb)



### Sequences

To train the model in the paper, run this command:

```train
python Sequences/main.py --dataset=SumTask --model=cgff --numEpochs=100
```



Main arguments to **Sequences/main.py** are:

Dataset arguments

```
  --dataDir dataDir         Data directory (to save basis) [default: data]
  --dataset dataset         One of [SumTask|Sum2Task|EvenMinusOddSumTask|GeometricDistributionTask]
  --nSamples nSamples       Number of samples [default: 10000]
```



Architecture/model arguments

```
  --model=model             Model: cgff
  --weightsAcrossDims       Different weights across different dimensions of the input.
  --penaltyAlpha=alpha      Penalty strength [default: 10]
  --penaltyMode=mode        L0 approximation (simple or sigmoid) [default: simple]
  --penaltyT=T              L0 approximation temperature [default: 1]
```

Other arguments include ``--batchSize``, ``--numEpochs``, ``—lr``, ``--seed`` with the usual meanings.





#### Sequence Example notebook
An example training of CGFF on the SumTask (extrapolation) is shown in [sequence_example.ipynb](sequence_example.ipynb)






## Results

### Images

Given 3 groups (rotation, color-permutation and vertical-flip), the table below shows **test extrapolation accuracy (%)** when the task is invariant to different subsets. Use  ``--groups=rotation_color_vflip`` and ``--dataset``, ``--datasetMode`` accordingly as given in the table. **Bold values** are significantly better (p-value < 0.05) than the baselines tested in the paper.

| I  (learn invariance to group G_I) | MNIST \{3, 4\} images (--dataset=mnistxtra) | MNIST all images (--dataset=mnistfullxtra) |
| ------------------------------------------------------------ | -----------------: | ------------: |
| {} (``—datasetMode='111'`` )                          | 94\.49±01\.49         | **90\.89±0\.93** |
| color (``—datasetMode='101'`` )                                | **94\.16±06\.43**     | **88\.69±2\.11** |
| rotation, vertical\-flip (``—datasetMode='010'`` )             | **95\.78±07\.11**     | **62\.68±6\.02** |
| rotation, vertical\-flip, color (``—datasetMode='000'`` )      | **94\.89±07\.49**     | 64\.99±2\.76     |



### Sequences

Given sequences of length $n$ and $\binom{n}{2}$ pairwise permutation groups of the form G_{i,j}={id, (ij)}, the table below shows **test extrapolation accuracy (%)** when the task is invariant to different subsets. Use ``--dataset`` accordingly as given in the table. **Bold values** are significantly better (p-value < 0.05) than the baselines tested in the paper.

| I  (learn invariance to group G_I) | Sequence Tasks   |
| ------------------------------------------------------------ | ----------------: |
| {} (``—dataset=GeometricDistributionTask`` )        | 95.70±03.05      |
| {(i, i+2k)}_{i,k} (``—dataset=EvenMinusOddSumTask`` ) | **71.85±26.61**  |
| {(i, j)}_{j>i\geq 2} (``—dataset=Sum2Task`` )            | **42.08±18.99**  |
| {(i, j)}_{j>i\geq 1}  (``—dataset=SumTask`` )            | **100.00±00.00** |














