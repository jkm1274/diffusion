# Financial Time Series Generation via Conditional Diffusion Models

> **Research conducted at Columbia University.** This repository provides an architectural and conceptual overview of the project. Due to proprietary data restrictions on the underlying benchmark indices, raw financial training data and downstream trading code are omitted from this public repository. Please refer to the [Research Report](./report.pdf) and [Bloomberg–Columbia Conference Poster](./poster.pdf) for full methodology and results.

🏆 **Selected for presentation at the Annual Bloomberg–Columbia Machine Learning in Finance Conference**

---

## Overview

Creating realistic synthetic financial data is critical for:

- Scaling machine learning models without overfitting to limited historical data
- Stress-testing portfolios under rare or extreme market conditions
- Augmenting sparse datasets across asset classes

Traditional generative methods struggle to capture the complex temporal patterns, non-linear dependencies, and heavy-tailed statistical distributions of real financial asset returns. This project applies a **Denoising Diffusion Probabilistic Model (DDPM)** framework — extended with classifier-free conditioning and alternative synthesizer architectures — to generate synthetic daily returns that statistically match benchmark indices across equities, fixed income, and real estate.

---

## Architecture

The framework is built around a two-phase diffusion process:

```
                    ┌──────────────────────┐
                    │   Real Return Data    │
                    │ (benchmark indices)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Forward Pass        │
                    │  (Add Gaussian Noise) │
                    │   T timesteps         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────────────────────────┐
                    │   Reverse Pass — Synthesizer Network       │
                    │   (Learns to predict & remove noise)       │
                    │                                            │
                    │   Conditioned on:                          │
                    │     • Timestep embeddings                  │
                    │     • Market regime labels  [6]            │
                    └──────────┬───────────────────────────────-┘
                               │
                    ┌──────────▼───────────┐
                    │  Synthetic Time       │
                    │  Series Output        │
                    └──────────────────────┘
```

### Key Technical Contributions

**1. Classifier-Free Conditional Diffusion**

Extends standard DDPM by encoding market regime labels alongside timestep embeddings. These conditional embeddings guide the reverse denoising process, allowing the model to dynamically adapt its outputs to distinct market regimes (e.g., high-volatility vs. low-volatility environments) without a separate classifier network.

**2. Alternative Synthesizer Architectures**

Rather than relying solely on a standard Transformer backbone, this project evaluates three advanced architectures:

| Architecture | Core Mechanism | Key Strength |
|---|---|---|
| **UniTST** | Slices sequences into localized patches; applies multi-variate attention | Reduced complexity; best MSE performance |
| **Perceiver** | Cross-attention bottleneck projecting high-dim inputs into a tight latent array | Highest training stability |
| **BVAE** (Bidirectional VAE) | Transforms inputs into latent representations | Most robust for learning baseline data distribution |

---

## Results

### Statistical Alignment vs. Tabular Baselines

The specialized sequence-to-sequence models vastly outperformed traditional tabular generators. The best-performing configuration — **UniTST with classifier-free conditional diffusion** — achieved the following reductions in Mean Squared Error (MSE) across key statistical moments:

| Metric | MSE Reduction |
|---|---|
| Mean | **−79.94%** |
| Standard Deviation | **−2.75%** |
| Skewness | **−70.00%** |
| Kurtosis | **−61.86%** |

### Architectural Trade-offs

- **BVAE** was the most robust at isolating and learning the baseline mean of original returns.
- **Perceiver** offered the most stable training dynamics across runs.
- **UniTST** delivered the best overall statistical fidelity to real index distributions.

### Open Challenge: Temporal Correlation

While all models successfully captured the distributional shape (moments) of index returns, accurately replicating long-term autocorrelation structure and cross-asset covariance remains an open problem — consistent with the broader financial time series generation literature.

---

## Tech Stack

| Component | Details |
|---|---|
| Diffusion framework | Custom DDPM with classifier-free guidance |
| Synthesizer architectures | UniTST, Perceiver, BVAE |
| Conditioning | Market regime label embeddings + timestep embeddings |
| Asset classes | Equities, fixed income, real estate (benchmark indices) |
| Training | PyTorch |

---

## Repository Contents

```
columbia-diffusion-finance/
├── report.pdf           # Full research report
├── poster.pdf           # Bloomberg–Columbia Conference poster
├── src/
│   ├── models/          # UniTST, Perceiver, BVAE synthesizer implementations
│   ├── diffusion/       # DDPM forward/reverse pass, noise schedulers
│   ├── conditioning/    # Classifier-free regime conditioning
│   └── evaluation/      # MSE, moment-matching, and ACF metrics
└── notebooks/
    └── results_viz.ipynb  # Reproduces key figures from the report
```

> Raw training data and downstream trading/backtesting code are excluded due to proprietary data restrictions on the underlying benchmark indices.

---

## References & Further Reading

- Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (NeurIPS Workshop 2021)
- Jaegle et al., *Perceiver: General Perception with Iterative Attention* (ICML 2021)
- Full citations available in the attached [Research Report](./report.pdf)
