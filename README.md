# 🎲 The Reparameterization Trick — Practice Repository

A hands-on, progressive set of exercises to deeply understand the
reparameterization trick and its role in training Variational Autoencoders.

## 🌐 Interactive Reference

An interactive HTML companion is deployed alongside this repository.

**[→ Open Interactive Explorer](https://reparametrization-trick.vercel.app)**

---

## 📁 Structure

```
reparam-trick-practice/
│
├── 01_basics/
│   ├── 01_sampling_problem.py       # Why naive sampling breaks gradients
│   ├── 02_score_function.py         # REINFORCE estimator (baseline comparison)
│   └── 03_reparam_estimator.py      # Reparameterization estimator
│
├── 02_gaussian_reparam/
│   ├── 01_manual_reparam.py         # Implement the trick from scratch w/ autograd
│   ├── 02_gradient_check.py         # Verify gradients numerically
│   └── 03_visualize_transform.py    # Visualize ε → z transformation
│
├── 03_vae_mnist/
│   ├── model.py                     # Encoder + Decoder architectures
│   ├── train.py                     # Training loop with ELBO loss
│   ├── evaluate.py                  # Reconstruction & latent space plots
│   └── generate.py                  # Sample new digits from prior
│
├── 04_experiments/
│   ├── variance_comparison.py       # REINFORCE vs Reparam variance
│   ├── latent_interpolation.py      # Interpolate between two images
│   └── disentanglement.py           # β-VAE experiment
│
├── utils/
│   ├── distributions.py             # Gaussian helpers
│   └── viz.py                       # Plotting utilities
│
└── requirements.txt
```

---

## 🚀 Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧭 Learning Path

Follow this order for the best understanding:

| Step | File | Concept |
|------|------|---------|
| 1 | `01_basics/01_sampling_problem.py` | Why stochastic nodes block gradients |
| 2 | `01_basics/02_score_function.py` | REINFORCE as a baseline |
| 3 | `01_basics/03_reparam_estimator.py` | The reparameterization estimator |
| 4 | `02_gaussian_reparam/01_manual_reparam.py` | Manual implementation + autograd |
| 5 | `02_gaussian_reparam/02_gradient_check.py` | Numerical gradient verification |
| 6 | `02_gaussian_reparam/03_visualize_transform.py` | Geometric intuition |
| 7 | `03_vae_mnist/model.py` → `train.py` | Full VAE on MNIST |
| 8 | `04_experiments/*` | Go deeper |

---

## 🧠 Core Idea (Quick Recap)

**Without the trick** — broken gradient flow:
```
z ~ 𝒩(μ, σ²)   ← stochastic node, ∂z/∂μ undefined
```

**With the trick** — clean gradient flow:
```
ε ~ 𝒩(0, 1)    ← fixed distribution, no parameters
z = μ + σ · ε   ← deterministic transform
∂z/∂μ = 1      ← trivial!
∂z/∂σ = ε      ← trivial!
```

---

## 📚 References

- **Kingma & Welling (2013)** — Auto-Encoding Variational Bayes [arXiv:1312.6114]
- **Rezende et al. (2014)** — Stochastic Backpropagation [arXiv:1401.4082]
- **Doersch (2016)** — Tutorial on Variational Autoencoders [arXiv:1606.05908]
# reparametrization_trick
