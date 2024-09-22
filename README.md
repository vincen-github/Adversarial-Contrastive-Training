# Unsupervised Transfer Learning via Adversarial Contrastive Training

Official repository of the paper **Unsupervised Transfer Learning via Adversarial Contrastive Training** 

Except for tuning λ for different dataset, all other hyperparameters of model structure and training policy used in our experiments are align with [Whitening for Self-Supervised Representation Learning](https://arxiv.org/abs/2007.06346). The implementation is all conducted in single NVIDIA Tesla V100 and checkpoints are stored in `data` each 100 epochs during training. All trained models are available in `model`.

## Supported Models
- ACT(Ours)
- Contrastive [SimCLR arXiv](https://arxiv.org/abs/2002.05709)
- BYOL [arXiv](https://arxiv.org/abs/2006.07733)
- WMSE [arXiv](https://arxiv.org/abs/2007.06346)

## Supported Datasets
- CIFAR-10 
- CIFAR-100
- Tiny ImageNet

## Results
| Method                                       | CIFAR-10 (linear) | CIFAR-10 (5-nn) | CIFAR-100 (linear) | CIFAR-100 (5-nn) | Tiny ImageNet (linear) | Tiny ImageNet (5-nn) |
|----------------------------------------------|-------------------|----------------|---------------------|------------------|------------------------|---------------------|
| [SimCLR](https://arxiv.org/abs/2002.05709)   | 91.80             | 88.42          | 66.83               | 56.56            | 48.84                  | 32.86               |
| [BYOL](https://arxiv.org/abs/2006.07733)     | 91.73             | 89.45          | 66.60               | 56.82            | **51.00**              | 36.24               |
| [W-MSE 2](https://arxiv.org/abs/2007.06346)  | 91.55             | 89.69          | 66.10               | 56.69            | 48.20                  | 34.16               |
| [W-MSE 4](https://arxiv.org/abs/2007.06346)  | 91.99             | 89.87          | 67.64               | 56.45            | 49.20                  | 35.44               |
| [ACT(Ours)]()                                         | **92.11**         | **90.01**      | **68.24**           | **58.35**        | 49.72                  | **36.40**           |


## Installation

The implementation is based on PyTorch. No uncommon package were used in our code.

#### Tiny ImageNet
If you want to reproduce our results, whatever for which SSL loss, You'd better acquire Tiny ImageNet through the script provided by [this repo](https://github.com/tjmoon0104/pytorch-tiny-imagenet). Otherwise the model hardly reached a top 1 accuracy of 1% at the end of training.

## Usage

Detailed settings are good by default, to see all options:
```
python -m train --help
python -m test --help
```

To reproduce the results from above table:
#### ACT
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128
```

#### WhiteningACT
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128 --method w_act
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128 --method w_act
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256 --method w_act
```

#### SIMCLR
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method contrastive
```

#### BYOL
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method byol
```

#### WMSE 2
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128 --method w_mse
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128 --method w_mse
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4 --method w_mse
```


#### WMSE 4
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
```

## Acknowledgement
This implementation is based on [htdt/self-supervised](https://github.com/htdt/self-supervised)

## Citation
```
```