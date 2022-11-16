## Implementation
- Have some experimental script for ImageNet or CIFAR-100 or MNIST.

## Experiences
#### Parameters and scaling
- Play with parameters (scale, strength of augmentation, outside manifold...)
- Try automatic sizing of hyperparameters.
- Experiment with growing `d`.

#### Architecture
- Try with layer norm instead of batch norm.

#### Grokking
- Define downstream task to check which eigenfunctions has been learned.
- Try on the half moon or on clustered data where harmonic are really clear.

#### Optimization
- Unbiased gradient for VIGReg regularizer.
