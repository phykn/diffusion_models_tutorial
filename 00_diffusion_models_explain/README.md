## Review: Expectation and Variance

### Expectation
In probability theory, expectation (also called expected value, mean, average) is a generalization of the weighted average.

### Definition
The expectation of a random variable $X$ is defined as  
\begin{aligned}
E[X] &= x_{1}p_{1} + x_{2}p_{2} + ... + x_{n}p_{n} \\
&= \sum_{i=1}^{n} x_{i}p_{i}
\end{aligned}  
where $x_{i}$ and $p_{i}$ are $i$th a possible outcome and its probability, respectively.

### Properties
1. $E[aX]=aE[X]$ where $a$ is a constant value.  
   $
   \begin{aligned}
   E[aX] &= \sum_{i=1}^{n} ax_{i}p_{i} \\
   &= a  \sum_{i=1}^{n} x_{i}p_{i} \\
   &= a  E[X]
   \end{aligned}
   $

2. $E[X+b]=E[X]+b$ where $b$ is a constant value.  
   $
   \begin{aligned}
   E[X+b] &= \sum_{i=1}^{n} (x_{i}+b)p_{i} \\
   &= \sum_{i=1}^{n} (x_{i}p_{i}+bp_{i}) \\
   &= \sum_{i=1}^{n} x_{i}p_{i} + \sum_{i=1}^{n} bp_{i} \\
   &= E[X] + b \sum_{i=1}^{n} p_{i} \\
   &= E[X] + b
   \end{aligned}
   $  
   We used the property that sum of all probability is equal to 1 $(\sum_{i=1}^{n} p_{i}=1)$.

3. $E[X+Y]=E[X]+E[Y]$    
   $
   \begin{aligned}
   E[X+Y] &= \sum_{i=1}^{n} (x_{i} + y_{i})p_{i} \\
   &= \sum_{i=1}^{n} (x_{i}p_{i} + y_{i}p_{i}) \\
   &= \sum_{i=1}^{n} x_{i}p_{i} + \sum_{i=1}^{n}y_{i}p_{i} \\
   &= E[X] + E[Y]
   \end{aligned}
   $

## References
1. [Diffusion models explained. How does OpenAI's GLIDE work?](https://youtu.be/344w5h24-h8) (
AI Coffee Break with Letitia, 2022)
2. [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://youtu.be/J87hffSMB60) (
AI Coffee Break with Letitia, 2022)
3. [Diffusion Models | Paper Explanation | Math Explained](https://youtu.be/HoKDTa5jHvg) (Outlier, 2022)
4. [DDPM - Diffusion Models Beat GANs on Image Synthesis (Machine Learning Research Paper Explained)](https://youtu.be/W-O7AZNzbzQ) (Yannic Kilcher, 2021)
5. [Diffusion models from scratch in PyTorch](https://youtu.be/a4Yfz2FxXiY) (DeepFindr, 2022)
6. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score) (Yang Song, 2021)
7. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) (Lilian Weng, 2021)
8. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Jonathan Ho, Ajay Jain, Pieter Abbeel, 2020)
