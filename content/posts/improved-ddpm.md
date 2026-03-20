---
title: "Diffusion Models - Part 2: Improved DDPM"
date: 2026-03-12T00:00:00Z
draft: false
tags: ["diffusion-models", "image-generation"]
math: true
ShowToc: false
---

In [Part 1 of the Diffusion Models series](https://halannhile.github.io/posts/ddpm/), I covered the theory behind DDPM, the most basic diffusion model, which consists of: 

* a forward process that gradually corrupts an image with Gaussian noise,
* a reverse process where a neural network U-Net learns to denoise step by step.

If you got through all the math needed to understand DDPM, were not scared by it, and are in fact even more fascinated about diffusion models, the next (and hopefully easier to digest) paper is: 

**Improved Denoising Diffusion Probabilistic Models** ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)), as the name suggests - is an improved version of DDPM, follows that framework, but improves it in 3 main ways: 

1. a **cosine noise schedule** that uses timesteps more effectively (for which I have actually covered both the theory and code in Part 1 - since I found it to be a marginal change that I could easily implement in my DDPM pipeline),

2. **learned reverse-step variance** that boosts the log-likelihood and sample quality,

3. a **hybrid loss** that combines both the MSE loss we saw in DDPM, and the negative variational lower bound (VLB) loss.

This blog post will be short since we have covered the bulk of the theory in part 1!   

---

# Table of Contents

[Section 1: Theory](#section-1-theory)
1. [Motivations for the Improved DDPM paper](#1-motivations-for-the-improved-ddpm-paper)
2. [The cosine schedule](#2-the-cosine-schedule)
3. [Learned variance in the reverse process](#3-learned-variance-in-the-reverse-process)
4. [Hybrid training objective](#4-hybrid-training-objective)

[Section 2: Code](#section-2-code)

[Section 3: Recap & What's Next](#section-3-recap--whats-next)

[Useful Resources](#useful-resources)

[Citation](#citation)

---

# Section 1: Theory 

## 1. Motivations for the Improved DDPM paper

Improved DDPM was motivated by **two limitations** of the original DDPM paper, which the DDPM authors themselves acknowledged: 

{{< sidenote >}}
$^1$**What are FID and NLL?**  

FID (Fréchet Inception Distance) and NLL (Negative Log-Likelihood) are distinct metrics:

- FID measures perceptual similarity to real data (lower is better) → FID aligns more with human visual judgment.
- NLL measures how well the model predicts the data distribution (lower is better) → NLL reflects probabilistic fit.
{{< /sidenote >}}

1. **Shortcoming #1 (likelihood gap)**: Despite producing high-quality samples, **DDPM failed to achieve competitive log likelihoods** compared to other likelihood-based generative models (VAEs, autoregressive models, etc.). The authors largely evaluated DDPM with sample-quality metrics such as FID rather than NLL$^1$, so the likelihood question remains open.

 
2. **Shortcoming #2 (slow sampling / inefficient steps)**: Generating a single sample from DDPM requires hundreds of sequential forward passes through the network - i.e. **the image generation process is too slow for practical use**, and many late timesteps seem to do very little useful work. 

In response, **Improved DDPM introduces two main techniques**, each targeting one of these shortcomings:

**1. Learned reverse-step variance + hybrid loss (for Shortcoming #1):**  

* DDPM authors tried fixing the reverse variance to either $\sigma^2\_t = \beta\_t$ *(the forward process variance)* or $\sigma^2\_t = \tilde{\beta}\_t$ *(the true posterior variance, a lower bound corresponding to $x_0$ being a delta function)* for the reverse-step variance and found similar sample quality either way. So they concluded the choice didn't matter much and simply kept the variance fixed.

* Improved DDPM treats this as a **missed opportunity for likelihood**. It parameterizes the model variance $\Sigma\_\theta(x\_t, t)$ and trains it with a small VLB term$^2$ added to the usual noise-prediction loss, with extra weight on early timesteps where the VLB contributes most.

* This directly improves the variational bound and therefore **improves NLL**, while keeping or slightly improving FID.

{{< sidenote >}}
**$^2$What is VLB?**

**variational lower bound (VLB)**  is a tractable lower bound on the log-likelihood $\log p\_\theta(x\_0)$, which is what we ultimately want to maximize but can't compute directly. It decomposes into a sum of per-step KL divergence terms - one for each reverse step - measuring how well the estimated $p\_\theta(x\_{t-1}|x\_t)$ matches the true posterior $q(x\_{t-1}|x\_t, x\_0)$. Maximizing the VLB is equivalent to minimizing those KL terms, pushing the learned reverse process as close as possible to the true (but intractable) reverse distribution.
{{< /sidenote >}}

**2. Cosine schedule (for Shortcoming #2, partially).**  

* With the linear schedule, the image becomes almost pure noise well before $t = T$, so many late timesteps contribute little information. 

* Improved DDPM proposes a **cosine schedule** for $\bar{\alpha}\_t$ that decays more smoothly, so each timestep sees a more useful noise level. They show that with this schedule, **skipping around 20% of reverse steps barely changes FID**, meaning compute is used more efficiently across timesteps. 

* This does **not fully solve** the slow-sampling problem (we still use a long chain), but it makes the existing steps more informative and improves sample quality at the *same* number of steps.

Together, these changes yield: 

1. better NLL (mainly from learned variance + hybrid loss),
2. better FID/sample quality at the *same* number of diffusion steps (from both the cosine schedule and better-trained variance).

## 2. The cosine schedule

You can find [detailed explanations for the cosine schedule](https://halannhile.github.io/posts/ddpm/#13-the-cosine-beta-schedule) in Part 1 (DDPM).

{{< figure align=center src="/images/improved-ddpm-fig3.png" alt="DDPM" title="Speed of adding noise using a linear vs. cosine schedule" caption="[Image source: Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)" width="60%" >}}

**TL;DR: The cosine schedule adds noise more slowly and evenly**. As a consequence, the "amount of denoising" is spread more evenly across timesteps. The model sees a wider range of noise levels during training instead of many steps that are already pure noise. Training stability and sample quality >> those of a linear schedule. 

## 3. Learned variance in the reverse process 

In DDPM, the reverse step is a Gaussian whose **mean** $\mu\_\theta(x\_t, t)$ is given by the network (via the predicted noise $\varepsilon\_\theta$), and whose **variance** is fixed to the true posterior variance $\tilde{\beta}\_t$:
 
$$p\_\theta(x\_{t-1}|x\_t) = \mathcal{N}(x\_{t-1}; \mu\_\theta(x\_t,t), \tilde{\beta}\_t I)$$
 
Recall from [Part 1](https://halannhile.github.io/posts/ddpm/#32-the-tractable-distribution-of-x_t-1) that the true posterior $q(x\_{t-1}|x\_t, x\_0)$ is tractable when conditioned on $x\_0$, and is a Gaussian with:
 
$$\tilde{\mu}\_t(x\_t, x\_0) = \frac{\sqrt{\bar{\alpha}\_{t-1}} \cdot \beta\_t}{1 - \bar{\alpha}\_t} x\_0 + \frac{\sqrt{\alpha\_t} \cdot (1 - \bar{\alpha}\_{t-1})}{1 - \bar{\alpha}\_t} x\_t$$
 
$$\tilde{\beta}\_t = \frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t$$

DDPM fixes the reverse-step variance $\tilde{\beta}_t$ and does not learn it. 

Improved DDPM instead **parameterizes the variance and learns it**. Specifically: 


{{< sidenote >}}

$^3$ 
* This is a **geometric** (log-linear) interpoloation, not an arithmetic one.

* The vector $v$ is an output of the network and thus depends on both $x_t$ and $t$.

{{< /sidenote >}}

* It parameterizes $\Sigma\_\theta(x\_t, t)$ as a **log-domain interpolation** between $\beta_t$ and $\tilde{\beta}_t$ - i.e., the upper and lower bounds on the reverse-step variance. The network outputs a vector $v$ (one component per dimension), and the variance$^3$ is: 

$$\boxed{\Sigma\_\theta(x\_t, t) = \exp\bigl(v \log \beta\_t + (1 - v) \log \tilde{\beta}\_t\bigr)}$$

At first, I took this formulation for granted and just agreed with what the authors proposed here. But after learning more about it, I realized how and why they were able to come up with this:

* $\Sigma\_\theta(x\_t, t)$ need to stay within the range $[\beta_t$, $\tilde{\beta}_t]$, which is known to be the reasonable interval for the reverse-step variance. However, this range is very small, making direct prediction unstable. 

* Hence, the authors formulated it such that the network needs to predict/learn an unconstrained interpolation weight $v$, and map it to that range via a geometric interpolation in the log-space, which naturally keeps the output between the two bounds regardless of what $v$ the network outputs.

Now that we understand how $\Sigma\_\theta(x\_t, t)$ is formulated, the next question is, **how do we train/learn it?**

* We need a loss that is sensitive to the specific choice of variance. $L_{simple}$ (the noise-prediction loss) is not, since it only depends on the predicted mean. 

* Hence, the VLB is the right signal because it directly measures, at each timestep $t$, how close the model's reverse step $p\_\theta(x\_{t-1}|x\_t)$ is to the true posterior $q(x\_{t-1}|x\_t, x\_0)$. And this *closeness* depends on both the mean *and* variance.


{{< sidenote >}}
$^4$Don't worry too much about the math, I just want to leave it here for reference, and for the readers to understand why we use Gaussian noise in diffusion models - it's because of nice properties such as this one here. [This blog post](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/) is a good reference if you're interested in knowing how the KL divergence for 2 Gaussians is derived. 
{{< /sidenote >}}

Since both distributions are Gaussian, the KL divergence has a closed form$^4$:

$$ \mathrm{KL}\bigl( q(x_{t-1}\mid x_t,x_0) \,\big\|\, p_\theta(x_{t-1}\mid x_t) \bigr)
= \tfrac{1}{2} \left[ \log \tfrac{\Sigma_\theta(x_t,t)}{\tilde{\beta}\_t} + \tfrac{\tilde{\beta}\_t + \|\tilde{\mu}\_t(x_t,x\_0) - \mu\_\theta(x_t,t)\|^2}{\Sigma\_\theta(x_t,t)} - 1 \right] $$

**The better $\Sigma\_\theta$ matches the true posterior’s variance (or spread) at each step, the smaller this KL term and the tighter the VLB.**

**We've talked about the variance, what about the mean?** In principle, the same VLB term could also be used to update $\mu_\theta$, but the paper deliberately keeps the two separated: 

* a **stop-gradient** is applied to $\mu_\theta$ inside the VLB term, so gradients from the VLB only flow into $\Sigma_\theta$, which $\mu_\theta$ is trained solely by $L_{simple}$

* this separation is important for stability - more on it in Section 4. 

### TL;DR: Why do we need to learn the variance instead of keeping it fixed? 

* A fixed $\tilde{\beta}_t$ is a one-size-fits-all choice: the same variance is used at every step regardless of what the model actually needs. 

* Learned variance lets the model adapt. It can use a smaller variance when it's confident about the denoising direction, and a larger one when uncertain. 

* In general, this tighter fit to the data improves the log-likelihood and in practice often improves sample quality (measured by FID) as well. 

## 4. Hybrid training objective  

Recall in [this section from Part 1](https://halannhile.github.io/posts/ddpm/#4-the-training-objective) how DDPM training uses a simple loss: 

$$\mathcal{L}\_{\text{simple}} = \mathbb{E}[\| \varepsilon - \varepsilon\_\theta(x\_t, t) \|^2]$$

which corresponds to predicting the noise, and this loss actually gives good samples even though it's not the full VLB. 

**Improved DDPM keeps the simple loss $L_{simple}$ for the *mean* $\mu_\theta$ (noise prediction), and adds the VLB so that the *variance* $\Sigma_\theta$ is also trained with the proper variational objective.**

The total loss is a **hybrid loss**: 

$$\boxed{\mathcal{L} = \mathcal{L}\_{\text{simple}} + \lambda \mathcal{L}\_{\text{vlb}}}$$

Here:

* $\mathcal{L}\_{\text{vlb}}$ is the negative variational lower bound, so we **minimize $\mathcal{L}\_{\text{vlb}}$**.

* The scalar $\lambda$ is small (e.g., 0.001) so that the main signal still comes from $\mathcal{L}_{simple}$ and training remains stable. 

The paper defines $\mathcal{L}\_{\text{vlb}}$ explicitly as a sum of per-step terms:
 
$$\mathcal{L}\_{\text{vlb}} = L\_0 + L\_1 + \cdots + L\_{T-1} + L\_T$$
 
where:
 
$$L\_0 = -\log p\_\theta(x\_0 | x\_1)$$
 
$$L\_{t-1} = D\_{\text{KL}}\bigl[ q(x\_{t-1}|x\_t, x\_0) | p\_\theta(x\_{t-1}|x\_t)\bigr] \quad \text{for } t = 1, \ldots, T$$
 
$$L\_T = D\_{\text{KL}}\bigl[q(x\_T|x\_0) | p(x\_T)\bigr]$$
 
- $L\_0$ is the **reconstruction term** - how well the model recovers the clean image $x\_0$ from $x\_1$ (the slightly noised version). It is the only term that directly measures image reconstruction quality.
- $L\_{t-1}$ is the **denoising term** at each step - the KL divergence between the true posterior $q(x\_{t-1}|x\_t, x\_0)$ and the model's reverse step $p\_\theta(x\_{t-1}|x\_t)$. Since both are Gaussians, this has a closed form (see above). This is the term that provides the training signal for $\Sigma\_\theta$.
- $L\_T$ is the **prior matching term** - how close the fully noised $x\_T$ is to a standard Gaussian $\mathcal{N}(0, I)$. This does not depend on $\theta$ at all (the forward process is fixed), so it is a constant during training and can be ignored for optimization.

During backprop, gradients from $\mathcal{L}\_{\text{vlb}}$ would normally flow through both $\Sigma\_\theta$ and $\mu\_\theta$, essentially updating both. But the paper applies a **stop-gradient to $\mu\_\theta$** inside $\mathcal{L}\_{\text{vlb}}$, which means: when computing the VLB loss, the network's mean output $\mu\_\theta$ is treated as a fixed constant. Gradients are blocked from flowing through it. So $\mathcal{L}\_{\text{vlb}}$ can only update $\Sigma\_\theta$.
 
The reason behind this design choice is actually so interesting:

* Because $\mathcal{L}\_{\text{vlb}}$ and $\mathcal{L}\_{\text{simple}}$ would otherwise send competing gradient signals to $\mu\_\theta$ - the two losses are not perfectly aligned, and letting both update the mean simultaneously destabilizes training. 

* The clean separation, then, looks like this: $\mathcal{L}\_{\text{simple}}$ is solely responsible for teaching the network to predict noise (training $\mu\_\theta$), and $\mathcal{L}\_{\text{vlb}}$ is solely responsible for teaching the network what variance to use at each step (training $\Sigma\_\theta$).

---

# Section 2: Code 

## 1. Improved DDPM official PyTorch implementation

The codebase for Improved DDPM is made public by the authors at: https://github.com/openai/improved-diffusion 

It's very simple to navigate, so I highly recommend you check it out. 

## 2. My PyTorch implementation (coming soon)

If you've had the chance to look through my PyTorch implementation for DDPM [here](https://github.com/halannhile/ddpm), then these files will remain unchanged: `dataset.py`, `noise_scheduler.py`, `unet.py`, `utils.py`. 

The only files that need changing to implement new ideas proposed in Improved DDPM are: `diffusion.py`, `ddpm.py`, `sample.py` and `train.py` - and I've included either a suffix `_improved` or prefix `improved_` to these files to make it clear. 

---

# Section 3: Recap & What's Next 

I will update this section when I'm done with my PyTorch implementation for Improved DDPM and have some generated samples from that model to showcase here.

After that, I'm looking forward to diving into the paper **Denoising Diffusion Implicit Models (DDIM)** ([Song et al., 2021](https://arxiv.org/abs/2010.02502)), which dramatically speeds up sampling from 1000 steps to as few as 50. *So far, my experience with generating images from diffusion models has been painfully slow.*

---

# Useful Resources


### Theory 

* [Lil’Log - What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Jia-Bin Huang - How I Understand Diffusion Models](https://www.youtube.com/watch?v=i2qSxMVeVLI)

### Code 

* [Official Improved DDPM repo written in PyTorch](https://github.com/openai/improved-diffusion)

---

# Citation

```
Le, Nhi. "Diffusion Models - Part 2: Improved DDPM". halannhile.github.io (March 2026). https://halannhile.github.io/posts/improved-ddpm/
```

**BibTeX:**

```bibtex
@article{nhi2025ddpm,
  title = {Diffusion Models - Part 2: Improved DDPM},
  author = {Nhi},
  journal = {halannhile.github.io},
  year = {2026},
  month = {March},
  url = "https://halannhile.github.io/posts/improved-ddpm/"
}
```
