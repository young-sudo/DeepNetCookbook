# A Cookbook for Deep Neural Networks

*by Younginn Park*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F37626?style=for-the-badge&logo=google&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003366?style=for-the-badge&logo=plotly&logoColor=white)
![ClearML](https://img.shields.io/badge/ClearML-00BFA6?style=for-the-badge&logo=mlflow&logoColor=white) 

Repository contains solutions for selected Deep Neural Network architecture optimization tasks using **PyTorch** and fine-tuned for running in the Google Colab environment.

The implemented solutions were a part of the Deep Neural Network course at the University of Warsaw 2024/25, where they achieved a **top 10% class ranking** for performance.

Projects include methods, proposed in the latest publications, with a potential for improving existing models

>For more details and implementation visit each subproject's Jupyter Notebook

<div style="margin-top: 20px; margin-bottom: 20px;">
<details>
  <summary><font size="4"><b>Table of Contents</b></font></summary>
  
1. [Proximal Backpropagation (ProxyProp)](#1-proximal-backpropagation-proxyprop)
2. [Adversarial Training (AdvProp and SparseTopK)](#2-adversarial-training-advprop-and-sparsetopk)
3. [Differential Attention (Diff Attention)](#3-differential-attention-diff-attention)
4. [Proximal Policy Optimization (PPO) and Random Network Distillation (RND)](#4-proximal-policy-optimization-ppo-and-random-network-distillation-rnd)

</details>
</div>

# 1. **Proximal Backpropagation ([ProxyProp](https://arxiv.org/abs/1706.04638))**

Modification for the backpropagation algorithm by taking implicit instead of explicit gradient steps to update the network parameters during neural network training.

The update of weights is defined as:

$$
%\begin{equation}
W^{(l)} = \text{arg min}_{W} \frac{1}{2} || W \cdot g^{(l-1)} + b^{(l)} - f^{(l)}_{*} ||^2 + \frac{1}{2\eta} || W - W^{(l)} ||^2
%\end{equation}
$$

This proximal operator in many cases results in a descent that is quicker than in the case of traditional gradient descent. Interestingly enough, the proximal operator arose in the context of gradient descent in which the minimized function is not differentiable, and gradient descent cannot be applied directly as we cannot compute the Jacobi matrix.

## Recap of backpropagation model

### Forward pass

We begin with $g^{(0)} = x$.

To apply forward pass, use the following formulas L times:

$$
%\begin{equation}
f^{(l + 1)} = W^{(l+1)} g^{(l)} + b^{(l+1)}
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
g^{(l)} = \sigma(f^{(l)})
\nonumber
%\end{equation}
%\\
$$

$l = 1, 2, \ldots, L$.


Then, use output of the network is $g^{(L)}$ to compute the loss $L(y, g^{(L)})$.

### Backward pass

We begin with $\frac{\partial L}{\partial g^{(L)}}$, which can be computed directly.

To compute gradients, use the following formulas:

$$
%\begin{equation}
\frac{\partial L}{\partial f^{(l)}} = \frac{\partial L}{\partial g^{(l)}} \odot g^{(l)} \odot (1 - g^{(l)})
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
\frac{\partial L}{\partial g^{(l)}} = (W^{(l+1)})^{T} \cdot \frac{\partial L}{\partial f^{(l+1)}}
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial f^{(l)}}
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial f^{(l)}} \cdot g^{(l-1)}
\nonumber
%\end{equation}
$$

$l = L, L-1, \ldots, 1$.

## ProxProp similar to our backpropagation

The task is to implement an algorithm based on [ProxProp](https://arxiv.org/pdf/1706.04638v3). Refer to the section above or section 4.2 in [the paper](https://arxiv.org/pdf/1706.04638v3) for algorithm description. Here we present the overview of the algorithm to be implemented.

### Forward pass

Forward pass is the same as in backpropagation implementation used in this course:
We begin with $g^{(0)} = x$.

To apply forward pass, use the following formulas L times:

$$
%\begin{equation}
f^{(l + 1)} = W^{(l+1)} g^{(l)} + b^{(l+1)}
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
g^{(l)} = \sigma(f^{(l)})
\nonumber
%\end{equation}
%\\
$$

$l = 1, 2, \ldots, L$.


Then, use output of the network is $g^{(L)}$ to compute the loss $L(y, g^{(L)})$.

### Backward pass

We begin with $\frac{\partial L}{\partial g^{(L)}}$, which can be computed directly.

Gradients $\frac{\partial L}{\partial f^{(l)}}$, $\frac{\partial L}{\partial g^{(l)}}$ need to be computed.

$$
%\begin{equation}
\frac{\partial L}{\partial f^{(l)}} = \frac{\partial L}{\partial g^{(l)}} \odot g^{(l)} \odot (1 - g^{(l)})
\nonumber
%\end{equation}
$$
$$
%\begin{equation}
\frac{\partial L}{\partial g^{(l)}} = (W^{(l+1)})^{T} \cdot \frac{\partial L}{\partial f^{(l+1)}}
\nonumber
%\end{equation}
$$

For simplicity, biases will be updated as in standard backpropagation:
$$
%\begin{equation}
b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
\nonumber
%\end{equation}
$$
Remember to make the gradient update independent from the batch size by appropriate averaging!

Weights will be updated using values $f^{(l)}_{*}, g^{(l)}_{*}$. They are defined as follows:

$$
%\begin{equation}
g^{(L)}_{*} = g^{(L)} - \eta \frac{\partial L}{\partial g^{(L)}}
%\end{equation}
$$

$$
%\begin{equation}
f^{(l)}_{*} = f^{(l)} - \frac{\partial g^{(l)}}{\partial f^{(l)}} \cdot (g^{(l)} - g^{(l)}_{*})
%\end{equation}
$$

$l = 1, 2, \ldots, L$.

$$
%\begin{equation}
g^{(l)}_{*} = g^{(l)} - \frac{\partial f^{(l+1)}}{\partial g^{(l)}} \cdot (f^{(l+1)} - f^{(l+1)}_{*})
%\end{equation}
$$

$l = 1, 2, \ldots, L - 1$.

 The update of weights is defined as:

$$
%\begin{equation}
W^{(l)} = \text{arg min}_{W} \frac{1}{2} || W \cdot g^{(l-1)} + b^{(l)} - f^{(l)}_{*} ||^2 + \frac{1}{2\eta} || W - W^{(l)} ||^2
%\end{equation}
$$

$l = 1, 2, \ldots, L$.


# 2. **Adversarial Training ([AdvProp](https://arxiv.org/abs/1911.09665) and [SparseTopK](https://openreview.net/forum?id=QzcZb3fWmW))**

Improvement of robustness for image classification models by training on batches containing both clean and adversarially perturbed examples. This forces the model to learn features that are invariant to small, malicious input changes, leading to better generalization and resilience against attacks.

MiniImageNet dataset was used - a downscaled subset of [ISVLRC ImageNet-1k](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview), with only 10 classes (RGB, irregular sizes up to 256x256). These images were subject to various transforms in order to make the model robust against distortions and adversarial images, making the model rely less on texture or shape.

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/cats.png" alt="cats" width="500"/>
  </figure>
</div>

We will look at a modern residual convolutional net. While they perform very well on image classification tasks, some problems they commonly have are that:<br>
* they rely too much on small-scale features (textures) rather than large-scale ones (shape). This often generalizes poorly to unseen datasets and is less human-aligned (e.g. explanations of why a model chose this class may be less interpretable).
* they are very susceptible to adversarial images, i.e. inputs maliciously altered in a way that is imperceptible to humans and shouldn't change the classification, but completely fool the model, making it output high probabilities for unrelated classes.

Training on adversarial examples unfortunately tends to decrease accuracy on plain (unmodified) images a lot.<br>
The authors of [Adversarial Examples Improve Image Recognition](https://arxiv.org/abs/1911.09665) hypothesize that<br>
this is because adversarial examples (and the model activations they induce) follow different distributions.<br>
They propose addressing that by using auxilliary batch-norm-s for the adversarial images, which has been implemented here.

## SparseTopK

Another technique to improve robustness against style and pattern changes was proposed in
[Emergence of Shape Bias in CNNs through Activation Sparsity](https://openreview.net/forum?id=QzcZb3fWmW).

The idea is simple: in between some layers, enforce activation sparsity by zeroing out all but the top say 20% activations (by absolute value).
The hope is that the strong activations, which we keep, encode the more generalizable shape information.

More formally `SparseTopK`, for a fixed fraction $k$ like $20\%$, should be a module that for an input $x \in \mathbb{R}^{C \times H \times W}$ outputs:
$$ \begin{align*}
    x_{\text{out}}[c,h,w] &= x[c,h,w]\quad &&\text{ if } |x[c,h,w]| \geq \text{top-k-percentile}(x[c,:,:]) \\
                          &= 0 \quad &&\text{ otherwise}
\end{align*} $$


# 3. **Differential Attention ([Diff Attention](https://arxiv.org/abs/2410.05258))**

Enhancement for transformer attention by allowing the model to amplify attention to the relevant context while canceling noise by calculating attention scores as the difference between two separate softmax attention maps.

## The Input

**Input to the transformer model is a sequence of tokens.   
The length of the input sequence is bounded by model context size.**

**Tokens** can correspond to individual characters, words, or short character sequences.  
Tokens are usually represented as natural numbers.  
The **tokenizer** is a program that converts text to a sequence of tokens.  
For example, consider the word `habitat` if we apply a GPT-3 tokenizer to it, then we will get a sequence of three tokens `[5976, 270, 265]` that corresponds to `["hab", "it", "at"]`.  
The common approach is to tokenize the text using around 128k different tokens.  
You can read more about tokenizers [here](https://huggingface.co/docs/transformers/tokenizer_summary).

## The Output
For each input token, the transformer model outputs a probability distribution on the next token given the previous tokens in the context.  
For example, if the model inputs three tokens `["hab", "it", "at"]` for the token `it` the model will output probability distribution $\mathbb{P}(t_3 | t_1= hab, t_2=it)$.  
In general if we denote the function induced by the transformer network as $T$ and tokens as $t_i$ for $i\in\{1\dots n\}$, then $T(t_i)=\mathbb{P}(t_i|t_{i-1}\dots t_1)$, where the probability $\mathbb{P}$ is the distribution that the training data was sampled from.

## Transformer

Processing in the Transformer model starts with converting each input token to a vector of real numbers using the Embedding layer.  
The Embedding layer stores a matrix of shape `(num_tokens, hidden_dim)` where each row contains an embedding vector corresponding to a particular token.  
Initially, these vectors are random, but as the model trains it learns to associate `appropriate meaning` with each token.


After the embedding, the tokens (now vectors) are passed through the Transformer layers.
In each layer, two major sublayers are used
* Attention
* FeedForward

Input to those two sublayers is normalized, and a skip connection is applied. This skip connection takes the input to the sublayer before normalization
and adds it to the result of the sublayer.  

Attention allows each token (now represented as a vector) to look at the preceding tokens.
FeedForward is usually implemented as a variant of MLP and allows the model to transform the hidden representation of each token.

After processing, we linearly project each token to a vector of length `num_tokens` and use softmax to generate probability distribution on the next token.

Note that each token (or the corresponding vector) is processed by each layer independently except for the Attention layer, where tokens can "look" at other tokens.

<p align="center">
  <img  src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/transformer.png" alt="transformer" width=400>
</p>

### Attention

First, let's implement a Multihead attention mechanism.
The computation goes as follows.
* QKV computation:
    * The input $x$ of shape `(batch_size, sequence_lenght, hidden_dim)` (`(b, l, hid)` for short) gets multiplied by 3 weights: $W_K, W_Q, W_V$ to get matrices $Q, K, V$, each of size `(b, l, num_heads * head_dim)` (`(b, l, n_h * head)`). Often `n_h * head == hid`.
    * The $Q, K, V$ gets resized to `(b, l, n_h, head)` and then the dimensions get permuted to match `(b, n_h, l, head)`
* Causal Self-attention (shapes in paranthesis):
    * Attention pre-activation matrix $A'$ is calculated as $A' = QK^T$ `(b, n_h, l, l)`
    * $A'$ is normalized $A' = A' / \sqrt{head}$, this is to get entries of $A'$ with variance $1$ at initialization.
    * Causal mask is added $A' = A' + M$, `(b, n_h, l, l)` where $M$ is a causal mask, with 0 on and below main diagonal and $-\infty$ above.
    * Attention matrix $A$ is calculated $A = \text{softmax}(A')$ `(b, n_h, l, l)` , where `softmax` is taken row-wise (last dimension). Together with the step before it produces $A$ that is lower triangular and whose rows sum up to 1.
    * Per-head output gets calculated by $O' = AV$ `(b, n_h, l, head)`
* Ouput
    * Per-head outputs get concatenated by permuting dimensions of $O'$ to be `(b, l, n_h, head)` and then resizing to `(b, l, n_h * head)`.
    * $O'$ get's multiplied by output weight $W_O$ of size `(n_h * head, hid)` to get the output of the attention $O = W_O O'$ `(b, l, hid)`

### Differential Attention

Differential Attention works similarly to normal Attention, but with a few differences:
* Differential Attention partitions $Q$ and $K$ matrices into chunks: $Q_1$, $Q_2$, $K_1$, $K_2$. Each of them have a shape `(b, l, n_h, head / 2)`
* It calculates two attention matrices: $A_1$ and $A_2$ using respective $Q_i, K_i$ matrices. Note that here the attention pre-activation is normalized by $ \sqrt{head / 2}$. Causal mask is still added to the pre-activation.
* The final attention matrix is calculated as $A = A_1 - \lambda A_2$
* $\lambda$ in the equation above is calculated as $\lambda = \exp(\lambda_{K_1} * \lambda_{Q_1}) - \exp(\lambda_{K_2} * \lambda_{Q_2}) + \lambda_{\text{init}}$, where $\lambda_{K_1}, \lambda_{Q_1}, \lambda_{K_2}, \lambda_{Q_2}$ are all parameters of shape `head / 2`, $*$ denotes scalar multiplication of vectors and $\lambda_{\text{init}} = 0.8$
* Additionally, normalization is applied to per-head tokens $O' = \text{RMSNorm}(O') \cdot (1-\lambda_{init})$

## Nucleus Sampling and Temperature
Sometimes using the greedy approach is not the best solution (for example model may fixate on a specific topic or start to repeat itself).  
One of the methods to alleviate this is to use nucleus sampling along with appropriate softmax temperature.

## SwiGLUFeedForward
Lets start with implementing a [SwiGLU](https://arxiv.org/abs/2002.05202v1) feedforward layer.
Compared to the usual version of FeedForward, this version uses 3 Linear layers: $W_1, V: \mathbb{R}^{d_{hidden}} \rightarrow \mathbb{R}^{d_{inner}}$ and $W_2: \mathbb{R}^{d_{inner}} \rightarrow \mathbb{R}^{d_{hidden}}$.
The output of the layer on the input $x$ is produced in the following manner

$h_1 = W_1(x)$

$h_v = V(x)$

$h_2 = \text{SiLU}(h_1) \cdot h_v$

$\text{result} = W_2 (h_2)$

Where $\text{SiLU}(x) = \sigma(x)\cdot x$ (in the paper, SwiGLU uses Swish with $\beta = 1$, which is called SiLU in pytorch)

## Read More

You can read more about the Vanilla Transformer in the [original paper](https://arxiv.org/abs/1706.03762).
Two improvements to the Vanilla architecture are implemented - [SwiGLU](https://arxiv.org/abs/2002.05202v1) and [Differential Attention](https://arxiv.org/abs/2410.05258)


# 4. **Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) and Random Network Distillation ([RND](https://arxiv.org/abs/1810.12894))**

Improving exploration in reinforcement learning, enabling agents to learn more effectively in sparse reward environments in the context of [MiniHack](https://github.com/facebookresearch/minihack) environments. These environments are challenging procedurally-generated tasks based on the [NetHack Learning Environment (NLE)](https://github.com/heiner/nle), designed to test advanced reinforcement learning algorithms. While PPO is a robust and popular policy optimization algorithm, RND introduces intrinsic rewards to encourage exploration in sparse-reward environments.

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/nethack.png" alt="nethack" width="500"/>
    <br>
    <figcaption style="text-align:center;"><em>Roguelike game NetHack has been used to train the Agent</em></figcaption>
  </figure>
</div>

The combination of PPO and RND addresses two key challenges in reinforcement learning:
- <b>Efficient Policy Optimization</b>: PPO uses a clipped surrogate objective to ensure stable and efficient updates to the policy, preventing large deviations that could destabilize training.
- <b>Sparse Rewards</b>: Many environments provide limited feedback, making it difficult for agents to learn effective policies. RND helps by providing intrinsic rewards based on prediction errors of a randomly initialized network, encouraging exploration.

## Proximal Policy Optimization - PPO

Policy optimization methods in reinforcement learning face two major challenges: <b>sample inefficiency</b> and <b>training instability</b>. On-policy methods require fresh data collection for each policy update, making them sample inefficient since data cannot be reused once the policy changes. Additionally, early policy gradient approaches often suffered from instability during training, as large parameter updates could lead to performance collapse.

<b>Trust Region Policy Optimization (TRPO)</b> attempted to address these issues by introducing constraints on policy updates. However, TRPO's reliance on complex second-order optimization made it computationally expensive and difficult to implement. This led to the development of <b>Proximal Policy Optimization (PPO)</b>, which maintains TRPO's benefits while simplifying the implementation.

<p align="center">
<img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/ppo.png">
</p>

### Core Idea

The fundamental idea behind PPO is to prevent excessive policy changes during training by implementing a "clip" mechanism. This clipping function ensures that the ratio between new and old policies stays within a small range, typically $[1-\epsilon,1+\epsilon]$ where $\epsilon$ is usually between 0.1 and 0.2. When the policy update results in a ratio outside this range, the clipping function restricts the change, effectively creating a trust region without the computational complexity of second-order methods


PPO operates in two main phases: data collection and optimization. During data collection, the algorithm gathers trajectories using the current policy. In the optimization phase, it performs multiple epochs of minibatch updates on this collected data. $\pi_{\theta_{old}(a_t|s_t)}$ are outputs of the actor when we collected the data, $\pi_{\theta_{old}(a_t|s_t)}$ are the outputs of the policy after we performed minibatch update.

The PPO objective function is defined as:

$$ L^{CLIP}(\theta)=\hat {\mathbb{E}}_t \left [ \min(r_t(\theta) \hat A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t) \right ] $$

Here, $t$ indexes the time steps in the collected trajectories, $A_t$ denotes the estimated advantage function, and the probability ratio is defined as:

$$r_t(\theta) = {\pi_\theta(a_t|s_t) \over \pi_{\theta_{old}}(a_t|s_t)}$$

The hat over $A_t$ signifies that it is an **estimator** of the advantage function, and the hat over $E_t$ indicates an **empirical average** used to approximate the true expectation.

The objective function maximizes the expected return by increasing the likelihood of actions with positive advantages ($A_t > 0$) while decreasing the likelihood of actions with negative advantages ($A_t < 0$). This balanced approach facilitates stable policy improvement. The expectation operator $\mathbb{E}_t$ indicates averaging over the sampled trajectories during data collection.

### Implementation Details

1. <b>Actor</b>: The actor represents the policy, $\pi_\theta(a_t|s_t)$, which maps states to a probability distribution over actions. It is responsible for deciding what action the agent should take in a given state. During training, the actor is updated to maximize the PPO objective function by either increasing or decreasing the likelihood of actions based on their estimated advantages.

2. <b>Critic</b>: The critic estimates the value of a state, typically using a value function $V_{\phi}(s_t)$. This value function helps determine how "good" a particular state is by predicting the expected cumulative reward from that state onward. The critic is trained to minimize the mean squared error between its predictions and the actual discounted returns observed during training.

3. <b>Data Collection</b>: In this phase, trajectories are collected by having the agent interact with the environment using its current policy $\pi_{\theta_{old}}(a_t|s_t)$. Each trajectory contains sequences of states, actions, rewards, and log probabilities of actions under the current policy. These trajectories are used to compute several key quantities:
  - Discounted Returns: The cumulative rewards from each state onward.

 $$ G_t = \sum_{t=0}^{∞} \gamma^t r_t $$
 $$ G_t = \sum_{t=0}^k \gamma^t r_t + V_{\phi}(s_{k+1})  $$

  - Advantages: The difference between observed returns and predicted state values. PPO uses Generalized Advantage Estimation (GAE) to compute these advantages efficiently and reduce variance.
  
 $$ \hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^∞ (\gamma \lambda)^l \delta_{t+l}^V$$
 $$\delta_t^V=-V(s_t) + r_t + \gamma V(s_{t+1})$$

4. <b>Optimization</b>: Once data is collected, PPO performs multiple epochs of minibatch updates on this data to improve both the actor (policy network) and the critic (value network). The optimization process involves:
  - Policy Update (Actor): The PPO objective function is used to update the policy network.
  
 $$ L^{CLIP}(\theta)=\hat {\mathbb{E}}_t \left [ \min(r_t(\theta) \hat A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t) \right ] $$

  - Value Function Update (Critic): The critic network is updated by minimizing a mean squared error loss. Here, $G_t$ represents the discounted return or target value for state $s_t$
  
 $$L^{VF}=\frac{1}{2N}\sum_{t=1}^{N}(V_{\phi}(s_t)-G_t)^2$$
  
  - Entropy Regularization: To encourage exploration and prevent premature convergence to suboptimal policies, an entropy term is often added to the loss function.

 $$L^{ENTROPY}=-\beta \sum_{\alpha} \pi_{\theta}(a|s) \text{log} \pi_{\theta}(a|s)$$

### Training Loop
The overall training loop for PPO can be summarized as follows:
- Collect trajectories using the current policy.
- Compute discounted returns and advantages using GAE.
- Perform multiple epochs of minibatch updates on both:
  - The actor using the clipped surrogate objective.
  - The critic using mean squared error loss.
- Repeat until convergence or until a predefined number of iterations is reached.

### References

- PPO: [J. Schulman et al., "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/abs/1707.06347.pdf)
- TRPO: [Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. 2015.](http://proceedings.mlr.press/v37/schulman15.pdf)
- GAE: [Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation."](https://arxiv.org/abs/1506.02438)

## Random Network Distillation - RND

Exploitation versus exploration is a critical topic in Reinforcement Learning. We'd like the RL agent to find the best solution as fast as possible. However, in the meantime, committing to solutions too quickly without enough exploration sounds pretty bad, as it could lead to local minima or total failure. Modern RL algorithms that optimize for the best returns can achieve good exploitation quite efficiently, while exploration remains more like an open topic.

### Classic exploration strategies

As a quick recap, let's first go through several classic exploration algorithms.
- Epsilon-greedy: The agent does random exploration occasionally with probability and takes the optimal action most of the time with probability ($1 - \epsilon$).
- Entropy loss term: Add an entropy term $H(\pi(a|s))$ into the loss function, encouraging the policy to take diverse actions.

### The Hard Exploration problem

The “hard-exploration” problem refers to exploration in an environment with very sparse or even deceptive rewards. It is difficult because random exploration in such scenarios can rarely discover successful states or obtain meaningful feedback.

[Montezuma's Revenge](https://en.wikipedia.org/wiki/Montezuma%27s_Revenge_(video_game)) is a concrete example of the hard-exploration problem. It is one of a few challenging games in Atari for DRL to solve. Many papers use Montezuma's Revenge to benchmark their results.

### Intrinsic Rewards as Exploration Bonuses
One common approach to better exploration, especially for solving the hard-exploration problem, is to augment the environment reward with an additional bonus signal to encourage extra exploration. The policy is thus trained with a reward composed of two terms $r_t = r_t^e + \beta r_t^i$, where $\beta$ is a hyperparameter adjusting the balance between exploitation and exploration.

- $r_t^e$ is an extrinsic reward from the environment at time $t$, defined according to the task in hand.
- $r_t^i$ is an intrinsic exploration bonus at time $t$.

This intrinsic reward is somewhat inspired by intrinsic motivation in psychology (Oudeyer & Kaplan, 2008). Exploration driven by curiosity might be an important way for children to grow and learn. In other words, exploratory activities should be rewarding intrinsically in the human mind to encourage such behavior. The intrinsic rewards could be correlated with curiosity, surprise, familiarity with the state, and many other factors.

### Prediction based exploration
Here, we will focus on intrinsic exploration bonuses rewarded for the improvement of the agent's knowledge about the environment. The agent's familiarity with the environment dynamics can be estimated through a prediction model. This idea of using a prediction model to measure curiosity was actually proposed quite a long time ago (Schmidhuber, 1991).

Specifically, we will implement Random Network Distillation (RND; Burda, et al. 2018) which introduces a prediction task <i>independent</i> of the main task. The RND exploration bonus is defined as the error of a neural network $\hat{f}(s_t)$ predicting features of the observations given by a <i>fixed randomly initialized</i> neural network $f(s_t)$. The motivation is that given a new state, if similar states have been visited many times in the past, the prediction should be easier and thus have lower error. The exploration bonus is
$r_i = |\hat{f}(s_t; \theta) - f(s_t)|_2^2$

<p align="center">
<img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/rnd.png">
</p>

Two factors are important in RND experiments:

1. Non-episodic setting results in better exploration, especially when not using any extrinsic rewards. It means that the return is not truncated at “Game over” and intrinsic return can spread across multiple episodes.
2. Normalization is important since the scale of the reward is tricky to adjust given a random neural network as a prediction target. The intrinsic reward is normalized by division by a running estimate of the standard deviations of the intrinsic return.

The RND setup works well for resolving the hard-exploration problem. For example, maximizing the RND exploration bonus consistently finds more than half of the rooms in Montezuma's Revenge.
