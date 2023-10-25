# Neighborhood Gradient Clustering
This code is related to the paper titled, "Neighborhood Gradient Clustering: An Efficient Decentralized Learning Method for Non-IID Data Distributions", In arXiv preprint [arXiv:2209.14390, 2022.](https://arxiv.org/abs/2209.14390) 

### Abstract
Decentralized learning algorithms enable the training of deep learning models over large distributed datasets generated at different devices and locations, without the need for a central server. In practical scenarios, the distributed datasets can have significantly different data distributions across the agents. The current state-of-the-art decentralized algorithms mostly assume the data distributions to be Independent and Identically Distributed (IID). In this paper, we focus on improving decentralized learning over non-IID data distributions with minimal compute and memory overheads. We propose Neighborhood Gradient Clustering (NGC), a novel decentralized learning algorithm that modifies the local gradients of each agent using self- and cross-gradient information. Cross-gradients for a pair of neighboring agents are the derivatives of the model parameters of an agent with respect to the dataset of the other agent. In particular, the proposed method replaces the local gradients of the model with the weighted mean of the self-gradients, model-variant cross-gradients (derivatives of the received neighbors? model parameters with respect to the local dataset - computed locally), and data-variant cross-gradients (derivatives of the local model with respect to its neighbors? datasets - received through communication). The data-variant cross-gradients are aggregated through an additional communication round without breaking the privacy constraints of the decentralized setting. Further, we present CompNGC which is a compressed version of NGC that reduces the communication overhead by 32x by compressing the additional communication round for cross-gradients. We demonstrate the empirical convergence and efficiency of the proposed technique over non-IID data distributions sampled from the CIFAR-10 dataset on various model architectures and graph topologies. Our experiments demonstrate that NGC and CompNGC outperform the existing state-of-the-art (SoTA) decentralized learning algorithm over non-IID data by 1-5% with significantly less compute and memory requirements. Further, we also show that the proposed NGC method outperforms the baseline by 5-40% with no additional communication.  

# Available Models
* 5 layer CNN
* ResNet
* VGG11

# Requirements
* found in env.yml file

# Hyper-parameters
* --world_size   = total number of agents
* --graph        = graph topology (default ring)
* --neighbors    = number of neighbors per agent (default 2)
* --optimizer    = global optimizer i.e., [d-psgd, ngc, cga, compngc, compcga]
* --arch         = model to train
* --normtype     = type of normalization layer
* --dataset      = dataset to train
* --batch_size   = batch size for training
* --epochs       = total number of training epochs
* --lr           = learning rate
* --momentum     = momentum coefficient
* --qgm          = activates quasi-global momentum for ngc or cga 
* --nesterov     = activates nesterov momentum
* --weight_decay = weight decay
* --gamma        = averaging rate for gossip 
* --skew         = amount of skew in the data distribution; 0 = completely iid and 1 = completely non-iid
* --alpha        = NGC mixing weight (either 0 or 1)

# How to run?

test file for running different methods on 5 nodes ring topology
```
sh test.sh
```

experiments to generate figure 1 and figure 3
```
sh figure.sh
```

5 layer CNN with 5 agents undirected ring topology with NGC optimizer:
```
python trainer.py  --data-dir ../data   --lr 0.01  --batch-size 160  --world_size 5 --skew 1 --gamma 0.1 --normtype evonorm --optimizer ngc --epoch 100 --arch cganet --momentum 0.9 --alpha 1.0 --graph ring --neighbors 2 --nesterov
```

ResNet20 with 10 agents undirected ring topology with NGC optimizer:
```
python trainer.py  --data-dir ../data   --lr 0.01  --batch-size 320  --world_size 10 --skew 1 --gamma 0.1 --normtype evonorm --optimizer ngc --epoch 100 --arch resnet --depth 20 --momentum 0.9 --alpha 1.0 --graph ring --neighbors 2 --nesterov
```
