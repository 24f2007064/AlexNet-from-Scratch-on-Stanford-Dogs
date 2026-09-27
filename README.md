The TenCrop transformation produces 10 different 227 × 227 crops for each image, which are evaluated during testing.
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

## Model Structure

The model is implemented from scratch using PyTorch.

### Feature Extractor

The feature extractor consists of 5 convolutional layers:

1. Conv2D: 3 → 96 channels, kernel size 11, stride 4
   - Batch Normalization
   - ReLU
   - Max Pooling: 3 × 3, stride 2

2. Conv2D: 96 → 256 channels, kernel size 5, padding 2
   - Batch Normalization
   - ReLU
   - Max Pooling: 3 × 3, stride 2

3. Conv2D: 256 → 384 channels, kernel size 3, padding 1
   - ReLU

4. Conv2D: 384 → 384 channels, kernel size 3, padding 1
   - ReLU

5. Conv2D: 384 → 256 channels, kernel size 3, padding 1
   - ReLU
   - Max Pooling: 3 × 3, stride 2

### Classifier

The classifier consists of:

- Dropout: 0.5
- Linear: 256 × 6 × 6 → 4096
- ReLU
- Dropout: 0.5
- Linear: 4096 → 4096
- ReLU
- Linear: 4096 → Number of classes

The feature output is flattened before being passed to the classifier.

## Data Transformations

### Training Transformations

Training images are processed using:

- Resize to 256 × 256
- Random Crop to 227 × 227
- Random Horizontal Flip with probability 0.5
- Random Affine Transformation:
  - Rotation: ±15°
  - Translation: up to 10% horizontally and vertically
  - Scale: 0.9–1.1
- Convert to Tensor
- Normalize using:

  `mean = [0.485, 0.456, 0.406]`

  `std = [0.229, 0.224, 0.225]`

### Test Transformations

For testing:

- Resize to 256 × 256
- TenCrop to 227 × 227
- Convert each crop to a Tensor
- Stack the 10 crops
- Normalize using the same mean and standard deviation as the training data

