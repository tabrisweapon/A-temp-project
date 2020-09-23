
Our implementation is based on the [official TRADES implementation](https://github.com/yaodongyu/TRADES), which is supported by the PyTorch library. Several minor changes are made, incluing checkpoint saving, tensorboard utilization, and package arrangement.

## Setup

For wider networks like WideResNet-34-10, a certain amount of GPU memory space is needed. And generally speaking, TRADES consume much more resources than PGD due to its more complicated computation graph. Specifically, we conduct most of our experiments on the NVIDIA TITAN RTX 2080Ti GPU, which approximately take two days to train WideResNet-34-10 for 100 epochs on CIFAR10.

To support tensorboard for PyTorch, it is requried that the PyTorch version is at least higher than 2.1.0. The Dataset of CIFAR10 will be automaticly downloaded in a "data" directory under the home directory "~" once train_cifar10.py is executed.

## Adversarial training for networks of various width.

To train TRADES with WideResNet-34-10 λ=18 for example, run the following command:
```
python train_cifar10.py --adv-method=trades --beta=18 --width=10 --model-dir=an_exp_folder

```
This will reproduce the best result of 56.22% in Table 2. Mannually setting the --width and --beta parameters can reproduce other results in the table.

To train PGD with WideResNet-34-10 λ=2 for example, run the following command:
```
python train_cifar10.py --beta=2 --width=10 --model-dir=an_exp_folder

```
This will reproduce the results in Table 6 in the appendix.



## Testing againt various attacks
Running the following commands on learned models will reproduce the results in Table 5.

To test results under the attack of PGD 20 * 0.007, run the following command:

```
python test_cifar10.py --width=10 --model-dir=an_exp_folder/highest.pt

```

To test results under the attack of PGD 20 * 0.003, run the following command:

```
python test_cifar10.py --width=10 --step-size=0.003 --model-dir=an_exp_folder/highest.pt

```

To test results under the attack of CW 20 * 0.007, run the following command:

```
python test_cifar10.py --width=10 --attack-loss=cw --model-dir=an_exp_folder/highest.pt

```
To test results under the AutoAttack:

```
python auto_cifar10.py --width=10 --model-dir=an_exp_folder/highest.pt

```


## Inspecting the local Lipschitzness

This important change is made in the file of attacks/trades.py, we highlight the key codes (line 43-44) here for conveniences:

```
grad_norm = grad.clone().detach().view(batch_size, -1).norm(p=1, dim=1)
max_grad_norm = torch.max(grad_norm, max_grad_norm)

```
