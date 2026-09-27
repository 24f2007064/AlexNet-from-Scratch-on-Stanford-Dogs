# AlexNet from Scratch on Stanford Dogs

Implementation of AlexNet from scratch using PyTorch on the Stanford Dogs dataset.

## Model Structure

- 5 Convolutional layers
- ReLU activation
- Max Pooling
- Batch Normalization
- Fully Connected layers
- Dropout
- Final classification layer

## Training Configuration

- Optimizer: SGD + Momentum
- Momentum: 0.9
- Learning Rate: 0.01 / 0.001
- Weight Decay (L2): 0.0005
- Batch Size: 32
- Input Size: 227 × 227

## Experiments

| Optimizer | LR | Epochs | Loss | Train Acc | Test Acc | Regularization |
|---|---:|---:|---:|---:|---:|---|
| SGD + Momentum | 0.01 | 25 | 2.3876 | 42.47% | 33.48% | L2 = 0.0005 |
| SGD + Momentum | 0.001 | 35 | 2.7489 | 33.62% | 27.48% | L2 = 0.0005 |
| SGD + Momentum | 0.01 | 55 | 1.3225 | 74.54% | 44.14% | L2 = 0.0005 |

## Overfitting

The model shows a significant gap between training and test accuracy.

In the 55-epoch experiment:

- Training Accuracy: 74.54%
- Test Accuracy: 44.14%

This indicates that the model tends to overfit the training data.

The experiments also show that increasing training epochs improved training accuracy but did not result in the same improvement on the test set.

## Results

The final notebook includes a chart comparing the training and test performance across the experiments.