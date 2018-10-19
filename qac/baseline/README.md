# Convolutional Neural Network (CNN) Model

We use the static variant (CNN-static) of the CNN architecture proposed by Yoon Kim:

> Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. CoRR, abs/1408.5. https://doi.org/10.3115/v1/D14-1181

Our implementation is based on the implementation by Alexander Rakhlin: https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras

## Computational Environment

We use Keras with the Tensorflow backend.

```sh
conda env create -f environment-cnn.yml
source activate stackexchange-cnn
```

Our experiments were ran on `Tesla P100-PCIE-12GB` with NVIDIA CUDA 9.0.

## Hyperparameters

We performed a grid search for each community. The parameters subject to optimization:

   * Dropout after convolutional layer
   * The number and size of filters
   * The number of hidden layers before the last fully connected layer
   * Learning rate and decay

The following parameters were found to work best.

|                 | Dropout | Filter Sizes | Num. of Filters | Hidden Dims.  | Learning Rate | Decay |
|-----------------|---------|--------------|-----------------|---------------|---------------|-------|
| Cross Validated | 0.7     | (3,4,5)      | 50              | [50,50,50,50] | 0.001         | 0     |
| Unix            | 0.8     | (3,4,5)      | 50              | [50]          | 0.001         | 0     |
| Ask Ubuntu      | 0.8     | (3,4,5)      | 50              | [50]          | 0.001         | 0     |
| Super User      | 0.8     | (3,8)        | 50              | [50]          | 0.001         | 0     |
| Stack Overflow  | 0.6     | (3,4,5)      | 50              | [50]          | 0.0001        | 0     |

A few parameters were kept static for all communities:

| Parameter                   | Value      |
|-----------------------------|------------|
| Batch Size                  | 64         |
| Max. Epochs                 | 10         |
| Optimizer                   | Adam       |
| Sequence Length             | 400 tokens |
| Embedding Dims.             | 300        |
| Min. delta (early stopping) | 0.3        |
| Patience (epochs)           | 3          |

## Model Execution and Grid Search

To execute a model, specify the set of hyperparameters in `qac/baseline/baseline_cnn_cv.py`. To perform a grid search, specify multiple parameters (see example in `baseline_cnn_cv.py`).
Then, execute the model training:

```
python qac/baseline/baseline_cnn_cv.py --community <community>
```

After the training completes, summary information is saved to `output/cnn/<community>/<date_time>/`.
