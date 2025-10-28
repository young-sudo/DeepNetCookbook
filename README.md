# A Cookbook for Deep Neural Networks

*by Younginn Park*

Repository contains solutions for some selected Deep Neural Network architecture tasks using **PyTorch** and optimized for running in the Google Colab environment. The implemented solutions were a part of the Deep Neural Network course at the University of Warsaw 2024/25.

Projects include methods, proposed in the latest publications, with a potential for improving existing models:
1. **Proximal Backpropagation ([ProxyProp](https://arxiv.org/abs/1706.04638))** - modification for the backpropagation algorithm by taking implicit instead of explicit gradient steps to update the network parameters during neural network training.
2. **Adversarial Training ([AdvProp](https://arxiv.org/abs/1911.09665) and [SparseTopK](https://openreview.net/forum?id=QzcZb3fWmW))** - improvement of robustness for image classification models by training on batches containing both clean and adversarially perturbed examples. This forces the model to learn features that are invariant to small, malicious input changes, leading to better generalization and resilience against attacks.
3. **Differential Attention ([Diff Attention](https://arxiv.org/abs/2410.05258))** - enhancement for transformer attention by allowing the model to amplify attention to the relevant context while canceling noise by calculating attention scores as the difference between two separate softmax attention maps.
4. **Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) and Random Network Distillation ([RND](https://arxiv.org/abs/1810.12894))** - improving exploration in reinforcement learning, enabling agents to learn more effectively in sparse reward environments.
