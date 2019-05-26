# Repository info
This repository contains the implementation of training process described in "One-element Batch Training by Moving Window" which allows reproducing results achieved by CWAE, WAE or SWAE model on MNIST, Fashion MNIST and CIFAR-10 datasets.

<p align="center">
  <img src="/ffhq_interpolations_large.png"/>
</p>

# Contents of the repository
```
|-- src/ - folder with python code
|---- train_models.py - the starting point for experiments, contains more information about command line arguments
|---- experiments/
|------ training_process.py - implementation of the trainer with the implementation of One-element Batch Training by Moving Window algorithm
|---- architectures/ - files containing architectures used in the paper.
|---- evaluation/ - implementation of metrics used to evaluate and compare models
|-- results/ - directory that will be created in order to store the results of conducted experiments
```
# Conducting the experiments
== Conducting the experiments ==
In order to conduct experiments run train_models file in src directory

Below we provide description of required input parameters:
```
model: { "wae", "cwae", "swae" }
dataset_name: { "mnist", "fashion_mnist", "cifar10" }
epochs_count: positive integer value
latent_size: size of chosen model's latent dimmension
batch_size_n: N value from the paper
batch_size_k: K value from the algorithm from paper
```

Below are some important optional parameters:
```
optimizer: { "gd", "adam" } - default value is "gd" which stands for Gradient Descent Optimizer
learning_rate: float value defining learning rate for chosen optimizer, default value is 0.001
```

There are more arguments described in src/train_models.py file

Below we provide commands that can be used in order to reproduce all the results reported in the paper.
Following commands should be run in src/ as working directory:

For WAE model:
```
python train_models.py wae mnist 500 20 64 64 --learning_rate 0.01
python train_models.py wae mnist 500 20 64 32 --learning_rate 0.005
python train_models.py wae mnist 500 20 64 1 --learning_rate 0.0025

python train_models.py wae cifar10 500 64 64 64 --learning_rate 0.0025
python train_models.py wae cifar10 500 64 64 32 --learning_rate 0.005
python train_models.py wae cifar10 500 64 64 1 --learning_rate 0.001

python train_models.py wae fashion_mnist 75 8 64 64 --optimizer adam --learning_rate 0.001
python train_models.py wae fashion_mnist 75 8 64 32 --optimizer adam --learning_rate 0.001
python train_models.py wae fashion_mnist 75 8 64 1 --optimizer adam --learning_rate 0.00001
```

For CWAE:
```
python train_models.py cwae mnist 500 20 64 64 --learning_rate 0.01
python train_models.py cwae mnist 500 20 64 32 --learning_rate 0.0075
python train_models.py cwae mnist 500 20 64 1 --learning_rate 0.0025

python train_models.py cwae cifar10 500 64 64 64 --learning_rate 0.001
python train_models.py cwae cifar10 500 64 64 32 --learning_rate 0.0025
python train_models.py cwae cifar10 500 64 64 1 --learning_rate 0.001

python train_models.py cwae fashion_mnist 75 8 64 64 --optimizer adam --learning_rate 0.001
python train_models.py cwae fashion_mnist 75 8 64 32 --optimizer adam --learning_rate 0.001
python train_models.py cwae fashion_mnist 75 8 64 1 --optimizer adam --learning_rate 0.00001
```

For SWAE:
```
python train_models.py swae mnist 500 20 64 64 --learning_rate 0.01
python train_models.py swae mnist 500 20 64 32 --learning_rate 0.001
python train_models.py swae mnist 500 20 64 1 --learning_rate 0.0025

python train_models.py cwae cifar10 500 64 64 64 --learning_rate 0.01
python train_models.py cwae cifar10 500 64 64 32 --learning_rate 0.001
python train_models.py cwae cifar10 500 64 64 1 --learning_rate 0.001

python train_models.py cwae fashion_mnist 75 8 64 64 --optimizer adam --learning_rate 0.001
python train_models.py cwae fashion_mnist 75 8 64 32 --optimizer adam --learning_rate 0.001
python train_models.py cwae fashion_mnist 75 8 64 1 --optimizer adam --learning_rate 0.00001
```

# Environment
We created the repository using the following configuration:

python 3.7.2
tensorflow 1.13.1
numpy 1.16.2
matplotlib 3.0.3

# Additional links
To compute FID Score we used the code from:
- https://github.com/bioinf-jku/TTUR

Implementation of WAE is based on the code from:
- https://github.com/tolstikhin/wae

Implementation of SWAE is based on the code from:
- https://github.com/skolouri/swae

Implementation of CWAE is based on the code from:
- https://github.com/gmum/cwae

