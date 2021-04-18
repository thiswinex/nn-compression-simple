# nn-compression-simple
A simple NN compression tool using ADMM.

Support weight pruning, weight quantization and custom compression operator.



## Usage

Just

```
from admm import YOUR_COMPRESSION_TOOL
```



### Weight Pruning Example

Import:

```
from admm import ADMM_pruning
```



Instantiating the class:

```
admm = ADMM_pruning(model, update_interval=args.admm_update_interval, l1=args.admm_l1)
```



After `loss.backward()` , you should:

```
admm.loss_update(loss)
```



If you want to mask gradient while finetuning, use:

```
admm.grad_mask()
```



Use `admm.apply_projW()` and `admm.restoreW()` at the beginning and the end of each model evaluation to get evaluation of the pruned model. Like:

```
admm.apply_projW()
# Evaluate your model here
admm.restoreW()
```



Want to finished pruning iteration or want to start finetuning, use:

```
admm.apply_projW()
```

to prune model thoroughly.



### Custom compression operator

You need to implement a class that inherits from class ADMM. Use your own `update()` function to define your compression operator. In brief you need to **project** the weight parameters (or other parameters you want to compress) into your constraint space.

For example, if you want to do pruning and quantization at the same time, you can simply call both update function one after other, which can project the weights to the intersection space of their constraint space.



### How it works

- [Zhang, Tianyun, et al. "A systematic dnn weight pruning framework using alternating direction method of multipliers." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.](https://openaccess.thecvf.com/content_ECCV_2018/html/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.html)

- [Leng, Cong, et al. "Extremely low bit neural network: Squeeze the last bit out with admm." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 32. No. 1. 2018.](https://ojs.aaai.org/index.php/AAAI/article/view/11713)

- [Ren, Ao, et al. "Admm-nn: An algorithm-hardware co-design framework of dnns using alternating direction methods of multipliers." *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*. 2019.](https://arxiv.org/pdf/1812.11677.pdf)

