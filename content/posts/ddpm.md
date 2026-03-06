---
title: "Diffusion Models - Part 1: DDPM"
date: 2025-03-06T00:00:00Z
draft: true
tags: ["diffusion-models", "image-generation"]
math: true
---

One of the places my curiosity took me to recently is Diffusion Models. Let's start with **Denoising Diffusion Probabilistic Models** (**DDPM**; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)).

### Useful Resources

* [Lil’Log - What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Jia-Bin Huang - How I Understand Diffusion Models](https://www.youtube.com/watch?v=i2qSxMVeVLI)




---

## Table of Contents

[Section 1: Theory](#section-1-theory)
1. [Overview of DDPM](#1-overview-of-ddpm)
2. [The forward process](#2-the-forward-process)
3. [The reverse process](#3-the-reverse-process)
4. [The training objective](#4-the-training-objective)
5. [Training & sampling algorithms](#5-training--sampling-algorithms)

[Section 2: Code](#section-2-code)

---

# Section 1: Theory 

## 1. Overview of DDPM


Prior to diffusion models, GANs largely dominated the conversation around high-fidelity image generation. Diffusion models, made practical and popular by DDPM, changed that narrative and have since become the dominant paradigm in modern image generation. 

The core idea behind diffusion models is surprisingly elegant: *train a neural network to gradually denoise an image corrupted with (Gaussian) noise.*

In other words, if we gradually destroy an image by adding noise over hundreds of steps until it becomes pure noise - and we train a model to gradually undo that process - we end up with a model that can generate images from scratch, starting from random noise. Diffusion models typically use *Gaussian noise* because it makes the forward and reverse processes mathematically tractable, and the training stable.

### The two processes of DDPM 

DDPM is built on two complementary processes: 

**1. Forward process** (also called the forward diffusion process): starting with a clean image $x\_0$, we systematically corrupt it over $T$ timesetps by adding small amounts of Gaussian noise. At the end of the diffusion process, the image becomes pure Gaussian noise $\mathcal{N}(0, I)$.

**2. Reverse process** (also called the reverse diffusion process): we start from pure noise and learn to denoise step-by-step until a clean image emerges. This is what the core architecture, the U-Net neural networks, learns to do. 

At generation time, we sample random noisy image $x\_T \sim \mathcal{N}(0, I)$ and run the reverse process to produce a new, clean image. 

## 2. The forward process

### 2.1. The original form

At each time step $t$, a small amount of Gaussian noise is added to the previous image $x\_{t-1}$ to give us a slighly more noisy image $x\_{t}$. The amount of noise is controlled by a **variance scheduler** $\beta\_{t}$ where $0 < \beta\_t < 1$. In practice the schedule is usually increasing: $0 < \beta\_1 < \beta\_2 < ... < \beta\_T < 1$.

$$q(x\_t | x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{1 - \beta\_t}\ x\_{t-1}, \beta\_t I)$$

In other words: to get $x\_t$, we scale down $x\_{t-1}$ slightly by $\sqrt{1 - \beta\_t}$ and add noise with variance $\beta\_t$.

### 2.2. The closed-form shortcut & reparameterization trick

Running the forward process step-by-step to get from $x_0$ to $x_t$ is painfully slow, since we need to generate all the intermediate steps sequentially $x\_{1}, x\_{2}, ..., x\_{t-1}$. Fortunately, the math works out so that we can jump directly from $x\_0$ to *any* $x\_t$ in a single step. I found this to be incredibly beautiful and elegant. 

*The full derivation is not included here for brevity. Please refer to my list of [useful resources](#useful-resources) for more details.*

Define: 

$$\alpha_t = 1 - \beta_t$$

$$\bar{\alpha}\_t = \prod\_{s=1}^{t} \alpha\_s$$

which is the cumulative product of all $\alpha$ values up to timestep $t$.

Then the closed-form forward process is: 

$$q(x\_t|x\_0) = \mathcal{N}(x\_t; \sqrt{\bar{\alpha}\_t}x\_0, (1-\bar{\alpha}\_t)I)$$

which we can sample using the **reparameterization trick** as:

$$\boxed{x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)} \quad (4)$$

This is **Equation 4** from the paper, and in my opinion it's the *single most important formula* for implementing DDPM. It means:
- When $t$ is small: $\bar{\alpha}\_t \approx 1$, so $x\_t \approx x\_0$ (barely noisy)
- When $t$ is large: $\bar{\alpha}\_t \approx 0$, so $x\_t \approx \varepsilon$ (pure noise)

This can be understood intuitively as follows: 

In $(4)$, $\sqrt{\bar{\alpha}\_t} x\_0$ is the **signal** and $\sqrt{1 - \bar{\alpha}\_t} \varepsilon$ is the **error**. The two coefficients always satisfy: 

$$\bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1$$

meaning signal and noise always sums to 1 in variance. As $t$ increases and $\bar{\alpha}_t$ decreases towards zero: the signal shrinks (the original image fades away) and the noise grows (Gaussian noise takes over). At $t = T$, $x_t \approx \epsilon \sim \mathcal{N}(0, 1)$, i.e., the image has been corrupted completely. 

**What exactly is the reparameterization trick?**

I exclude most mathematical derivations from this blog but since Equation (4) is so important in DDPM, I want to explain more about the reparameterization trick in Gaussian distribution. Let's first understand its basic statistical derivation. 

Say we have a random variable sampled from a distribution whose parameters (mean & variance) we want to optimize: 

$$x \sim \mathcal{N}(\mu, \sigma^2)$$

If we want to compute gradients wrt. $\mu$ and $\sigma$, we have a problem: we can't propagate through a sampling operation. Sampling is stochastic - it's not a deterministic function we can differentiate through. The gradient of "randomly draw a number" wrt. the distribution's parameters is undefined.

The trick works by separating the stochasticity from the parameters: Instead of sampling $x \sim \mathcal{N}(\mu, \sigma^2)$ directly, we do: 

$$\epsilon \sim \mathcal{N}(0, 1)$$

which is error sampled from a fixed, parameter-free distribution. Then:

$$x = \mu + \sigma \cdot \epsilon$$

This produces exactly the same distribution over $x$. But now the randomness is isolated in $\epsilon$, which has no parameters to differentiate through. The computations we need to perform flow cleanly through $\mu$ and $\sigma$ because the transformation $x = \mu + \sigma \cdot \epsilon$ is just arithmetic and fully differentiable. 

We've turned from "sampling from a parameterized distribution", to "sampling from a standard Gaussian $\mathcal{N}(0,1)$, then scale by $\sigma$ and shift by $\mu$".

In fact, the reparameterization trick, together with a key property of Gaussian distribution that **the sum of two independent Gaussians is also Gaussian** helps us iteratively derive $x_1$ from $x_0$, substitute into the derivation of $x_2$ from $x_1$ to get $x_0$ from $x_2$, so on and so forth, until we get to the final form in $(4)$. 

The parameterization trick also means $(4)$ can be understood as follows: to get from $x_t$ from $x_0$ is simply a shift-and-scale of an independent noise $\epsilon$.

**Why this trick matters for DDPM:**

Without it, for each training sample, we'd need to run $t$ sequential forward steps just to create the noisy input, and $t$ is randomly sampled up to $T$ each time (in DDPM, $T = 1000$). With the closed-form expression, creating $x_t$ from $x_0$ takes exactly one operation regardless of how large $t$ is. 

## 3. The reverse process

The reverse process learns how to undo all the corruptions we've made to the clean input images during the forward process.

What GANs attempt to do is to go from 100% noise to 0% noise in one shot, essentially asking the model to hallucinate an entire image from scratch with no *intermediate* guidance - this is notoriously unstable to train. 

With DDPM, we ask: given a noisy image $x_t$, what does the *slightly less noisy* version $x_{t-1}$ look like? If we could answer this for every $t$, we could start from pure noise $x_T$ and walk all the way back to a clean image $x_0$ in *incremental steps*. Taking many small steps rather than one giant leap is the exact same reason why the forward process adds noise gradually: small steps are predictable, large steps aren't. 

In fact, each denoising step is responsible for undoing roughly $1/T$ of the total noise. Because of this manageable and tractable denoising process, DDPM generates higher-quality images than GANs. 

### 3.1. The <ins>in</ins>tractable distribution of $x_{t-1}$

Unfortunately, the true reverse distribution $q(x_{t-1}|x_t)$ is intractable, since it requires knowing the true distribution of input images $p(x_0)$, which is the very thing we're trying to learn.

### 3.2. The tractable distribution of $x_{t-1}$

Fortunately, the *conditional* reverse distribution $q(x_{t-1}|x_t, x_0)$ - i.e., where we're given not only $x_t$ but also the original clean image $x_0$, is tractable. It's just a Gaussian: 

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

where the **posterior mean** is:

$$\tilde{\mu}\_t(x\_t, x\_0) = \frac{\sqrt{\bar{\alpha}\_{t-1}}\cdot \beta\_t}{1 - \bar{\alpha}\_t} x\_0 + \frac{\sqrt{\alpha\_t} \cdot (1 - \bar{\alpha}\_{t-1})}{1 - \bar{\alpha}\_t} x\_t$$

and the **posterior variance** is:

$$\tilde{\beta}\_t = \frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t$$

The math seems to come out of nowhere. But intuitively, the posterior mean is a weighted combination of 2 things: 

1. a contribution from the original image $x_0$
2. a contribution from the current noisy image $x_t$

Only one caveat: we don't know $x_0$ at inference/generation time! But that's the whole point of training a neural nets, so that we can **predict** $x_0$. During training, we have access to $x_0$ and hence the model can be supervised trained here. During inference, the model substitutes with its own prediction $\hat{x}_0$.

In fact, in the above formula: 

* the true posterior mean $\tilde{\mu}\_t(x\_t, x\_0)$ is intractable because it depends on $x_0$ which we don't have at inference

* the true posterior variance $\tilde{\beta}\_t$ is fully tractable since it depends only on 3 values , which are all fixed, precomputed constants from the noise schedule that require no neural network and no knowledge of $x_0$.

### 3.3. What the neural network learns: noise prediction (instead of clean image prediction)

The DDPM paper makes an interesting design choice: rather than training the model to predict $x_0$ directly, it's trained to predict the noise that was added to $x_0$ to create $x_t$. We can then "subtract" this predicted noise to get $\hat{x}_0$ (this is not a simple subtraction but has some scaling factor). 

This trained model is $\epsilon_{\theta}(x_t, t)$. 

Why? I asked myself the same question. Below are some motivating factors: 

1. predicting Gaussian noise in the same range $\mathcal{N}(0,1)$ is easier than predicting a full clean image from noisy input. 
2. at large timesteps where the image is noisy, asking the model "what was the original clean image?" is nearly impossible since there's little signal left. But asking "what Gaussian noise was added?" is more tractable since there's at least still some structure from Gaussian noise that we can partially infer. 
1. one neat thing is: predicting noise leads to a simple MSE loss (read more below). 

### 3.4. What does one denoising step look like? 

Once noise $\epsilon_{\theta}(x_t, t)$ is predicted, we can predict $\hat{x}_0$ via: 

$$\hat{x}\_0 = \frac{x\_t - \sqrt{1 - \bar{\alpha}\_t} \cdot \varepsilon\_\theta(x\_t, t)}{\sqrt{\bar{\alpha}\_t}}$$

We then plug this into the posterior mean $\mu_\theta(x_t,t)$ and simplify to get the sampling update: 

$$\boxed{\mu\_\theta(x\_t, t) = \frac{1}{\sqrt{\alpha\_t}} \left( x\_t - \frac{\beta\_t}{\sqrt{1 - \bar{\alpha}\_t}} \cdot \varepsilon\_\theta(x\_t, t) \right)}$$

In words, one denoising step works like this: 

1. take the current noisy image $x_t$, subtract out the predicted noise $\varepsilon\_\theta(x\_t, t)$ (scaled appropriately), and rescale, gives us the mean $\mu_\theta$ of the reverse distribution $p_\theta(x_{t_1}|x_t)$

2. to actually sample $x_{t-1}$, we then draw from a Gaussian distribution at that mean: 

$$\boxed{x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\tilde{\beta}_t} z, \quad z \sim \mathcal{N}(0,1)}$$ 

A very important note here is: after computing the mean $\mu_\theta$, we don't use it directly as $x_{t-1}$. Instead, we're adding back a small amount of fresh Gaussian noise, i.e., **scaled noise $\sqrt{\tilde{\beta}_t} z$**. 

**Why do we add noise back during sampling?**

This felt counterintuitive to me at first. We're trying to denoise, why add more? The reason is **diversity**, without which the model isn't truly generative. 

* If every reverse step were fully deterministic, then every starting noise $x_T$ would map to the same final image. The stochasticity at each step, via the added scaled noise, means that the reverse trajectories can diverge, allowing the model to generate diverse images. 

* The one exception is **the final step** ($t=1 \rightarrow t=0$) where we set $z=0$ and use the mean directly as the sampled image. At this point, adding noise would add random variation to the nearly-finished image, which would corrupt the fine details rather than add useful diversity. Hence, **the last step is fully deterministic**.

## 4. The training objective

In many blogs and videos that deep-dive into DDPM, the authors discuss this concept of the **ELBO** (Evidence Lower Bound), which involves a sum of KL divergences at each step. (Simply put, KL divergence is a measurement of how different one distribution is from another).

The math behind the ELBO can be terrifying to beginners. But luckily for us, the DDPM authors provide one key finding that the training objective can be simplified dramatically into a simple MSE loss: 

$$\boxed{L\_\text{simple} = \mathbb{E}\_{t, x\_0, \varepsilon} \left[ \| \varepsilon - \varepsilon\_\theta(\sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \varepsilon, t) \|^2 \right]}$$

This is **Equation 14** in the paper. In simple words, the training objective says that we:

* sample a random timestep
* add noise to the image 
* ask the model what noise was added 
* minimize the MSE between the true noise and the prediction 

## 5. Training & sampling algorithms

My PyTorch implementation of DDPM is built around these 2 algorithms from the paper: 

**Algorithm 1: Training**

```
repeat:
    x_0 ~ q(x_0)                              # sample a real image
    t ~ Uniform({1, ..., T})                  # sample a random timestep
    ε ~ 𝒩(0, I)                               # sample noise
    
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε         # corrupt image (closed-form forward)
    
    Take gradient step on:
        ∇_θ ‖ε - ε_θ(x_t, t)‖²               # MSE between true and predicted noise
until converged
```

**Algorith2 : Sampling (Reverse Diffusion)**

```
x_T ~ 𝒩(0, I)                                                     # start from pure noise
for t = T, T-1, ..., 1:
    z ~ 𝒩(0, I) if t > 1, else z = 0                              # no noise at final step
    x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t)) + √β̃_t·z   # denoise one step
return x_0                                                          # generated image                                                  
```

# Section 2: Code

In progress... I will update this section of the blog once my repo is on GitHub. 