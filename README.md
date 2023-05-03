## Diffusion Models Tutorial
Diffusion models in machine learning are a type of probabilistic generative model. Diffusion models are gaining attention due to their capacity to generate highly realistic images. It is also recognized for its exceptional performance in various fields such as text-to-image conversion, which converts text into images, image inpainting that replaces missing parts in an image, and super-resolution that enhances image quality. If you are interested in experimenting with diffusion models, you can try it out at https://stablediffusionweb.com

To comprehend diffusion models, one must familiarize themselves with several complex equations. Even I struggled with it, and honestly, I am still learning. Therefore, in this tutorial page, I aim to organize what I have studied in a more accessible manner. I hope it will be beneficial to those studying diffusion models.

This tutorial is divided into several parts. 
1. **Part 1** provides a summary of the background knowledge necessary for working with diffusion models. It is beneficial for those who require prior knowledge of concepts such as expected values and ELBO to refer to.
2. **Part 2** covers the fundamentals of diffusion models, including the forward and reverse process concepts. Additionally, we will implement diffusion models based on what we have learned so far in this section.
3. Diffusion models are known for their ability to generate high-quality images, but they require hundreds to thousands of samples, resulting in time-consuming problems. To tackle this issue, several methods have been proposed. **Part 3** of this tutorial focuses on DDIM, which was the initial approach proposed to address these problems.

In this tutorial, we will also introduce a more advanced sampling method.

## Table of contents
||Title|LINK|Update|
|---|---|---|---|
||**PART 1. Background**|
|1|Expectation and variance|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/01_expectation_and_variance.ipynb)|22.12.14|
|2|Reparameterization trick|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/02_reparameterization_trick.ipynb)|22.12.14|
|3|Kullback–Leibler divergence|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/03_kl_divergence.ipynb)|22.12.14|
|4|Evidence lower bound|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/04_elbo.ipynb)|22.12.14|
||**PART 2. Diffusion Models**|
|1|Forward and reverse process|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/05_forward_and_reverse.ipynb)|22.12.14|
|2|Noise schedule|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/06_noise_schedule.ipynb)|22.12.14|
|3|Example: DDPM|[LINK](https://github.com/phykn/diffusion_models_tutorial/tree/main/01_example/ddpm)|22.12.19|
||**PART 3. DDIM**|
|1|DDIM|[LINK](https://nbviewer.org/github/phykn/diffusion_models_tutorial/blob/main/notebooks/07_ddim.ipynb)|22.12.23|
|2|Example: DDIM|[LINK](https://github.com/phykn/diffusion_models_tutorial/tree/main/01_example/ddim)|22.12.23|

## References
1. [Expected value](https://en.wikipedia.org/wiki/Expected_value) (Wikipedia)
1. [Variance](https://en.wikipedia.org/wiki/Variance) (Wikipedia)
1. [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) (Wikipedia)
1. [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (Wikipedia)
1. [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) (Wikipedia)
1. [공돌이의 수학정리노트](https://angeloyeo.github.io)
1. [Matthew N. Bernstein](https://mbernste.github.io)
1. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score) (Yang Song, 2021)
1. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) (Lilian Weng, 2021)
1. [Diffusion models explained. How does OpenAI's GLIDE work?](https://youtu.be/344w5h24-h8) (
AI Coffee Break with Letitia, 2022)
1. [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://youtu.be/J87hffSMB60) (
AI Coffee Break with Letitia, 2022)
1. [Diffusion Models | Paper Explanation | Math Explained](https://youtu.be/HoKDTa5jHvg) (Outlier, 2022)
1. [DDPM - Diffusion Models Beat GANs on Image Synthesis (Machine Learning Research Paper Explained)](https://youtu.be/W-O7AZNzbzQ) (Yannic Kilcher, 2021)
1. [Diffusion models from scratch in PyTorch](https://youtu.be/a4Yfz2FxXiY) (DeepFindr, 2022)
1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Jonathan Ho, Ajay Jain, Pieter Abbeel, 2020)
1. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Prafulla Dhariwal, Alex Nichol, 2021)
1. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Alex Nichol, Prafulla Dhariwal, 2021)
1. [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796) (Ling Yang, et al., 2022)
1. [Diffusion Models in Vision: A Survey](https://arxiv.org/abs/2209.04747) (Florinel-Alin Croitoru, et al., 2022)
1. [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796) (Ling Yang, et al., 2022)
1. [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
1. [Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)