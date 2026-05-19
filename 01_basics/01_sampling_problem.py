"""
Exercise 01 — The Sampling Problem
===================================
Goal: Understand WHY naive sampling blocks gradient flow.

We want to minimize:
    f(μ) = 𝔼_{z ~ 𝒩(μ, 1)}[ z² ]

The true gradient is:  ∂f/∂μ = 2μ   (can be shown analytically)

Run this script to see that the naive gradient attempt fails,
while the reparameterized gradient works perfectly.
"""

import torch
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 1. Analytical answer (ground truth)
# ──────────────────────────────────────────────────────────────────────────────
# f(μ) = 𝔼[z²] where z ~ 𝒩(μ, 1)
# 𝔼[z²] = Var(z) + (𝔼[z])² = 1 + μ²
# ∂f/∂μ = 2μ

def true_gradient(mu: float) -> float:
    return 2 * mu


# ──────────────────────────────────────────────────────────────────────────────
# 2. Naive attempt — sampling directly, then calling .backward()
# ──────────────────────────────────────────────────────────────────────────────
def naive_gradient_attempt(mu_val: float, n_samples: int = 1000) -> float:
    """
    THIS WILL NOT WORK as intended.
    The sampling operation is not in the computation graph.
    PyTorch won't raise an error — it just silently gives wrong gradients (or None).
    """
    mu = torch.tensor([mu_val], requires_grad=True)

    # Naively sample from 𝒩(μ, 1)
    # NOTE: torch.normal() is NOT differentiable w.r.t. mean!
    z = torch.normal(mean=mu.expand(n_samples), std=1.0)  # z detached from graph

    loss = (z ** 2).mean()

    # Try to backprop
    try:
        loss.backward()
        grad = mu.grad.item() if mu.grad is not None else float('nan')
    except Exception as e:
        grad = float('nan')
        print(f"  Error: {e}")

    return grad


# ──────────────────────────────────────────────────────────────────────────────
# 3. Reparameterized gradient — CORRECT
# ──────────────────────────────────────────────────────────────────────────────
def reparam_gradient(mu_val: float, n_samples: int = 1000) -> float:
    """
    The reparameterization trick:
        ε ~ 𝒩(0, 1)   ← no parameters, just sample noise
        z = μ + ε      ← deterministic transform (σ=1 here)

    Now z is in the graph: ∂z/∂μ = 1  ✓
    """
    mu = torch.tensor([mu_val], requires_grad=True)

    # Sample from a FIXED distribution (no parameters)
    eps = torch.randn(n_samples)           # ε ~ 𝒩(0, 1)

    # Deterministic transform — this IS in the computation graph
    z = mu + eps                           # z = μ + ε

    loss = (z ** 2).mean()                 # f(z) = z²

    loss.backward()
    return mu.grad.item()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Compare
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mu_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    print("=" * 60)
    print(f"{'μ':>6} | {'True ∂f/∂μ':>12} | {'Naive':>12} | {'Reparam':>12}")
    print("-" * 60)

    for mu in mu_values:
        gt   = true_gradient(mu)
        naive = naive_gradient_attempt(mu)
        reparam = reparam_gradient(mu)
        naive_status = "✗ broken" if abs(naive - gt) > 0.5 else f"{naive:.4f}"
        print(f"{mu:>6.1f} | {gt:>12.4f} | {naive_status:>12} | {reparam:>12.4f}")

    print("=" * 60)
    print("\n✓ Reparameterized gradients match the true gradient closely.")
    print("✗ Naive sampling produces incorrect/missing gradients.\n")

    # ── Plot: Gradient estimates over μ ──
    mus = torch.linspace(-3, 3, 60)
    true_grads   = [true_gradient(m.item()) for m in mus]
    reparam_grads = [reparam_gradient(m.item(), n_samples=2000) for m in mus]

    plt.figure(figsize=(8, 4), facecolor='#0b0c10')
    ax = plt.gca()
    ax.set_facecolor('#13151b')

    ax.plot(mus.numpy(), true_grads,    color='#5df5a5', lw=2.5, label='True ∂f/∂μ = 2μ')
    ax.plot(mus.numpy(), reparam_grads, color='#5d9cf5', lw=2,   label='Reparam estimate', alpha=0.85, linestyle='--')

    ax.axhline(0, color='#2a2e3f', lw=1)
    ax.set_xlabel('μ', color='#717a9e')
    ax.set_ylabel('Gradient', color='#717a9e')
    ax.set_title('Reparameterized Gradient vs True Gradient', color='#e0e4f0', pad=12)
    ax.legend(facecolor='#1c1f2b', labelcolor='#e0e4f0', edgecolor='#2a2e3f')
    ax.tick_params(colors='#717a9e')
    for spine in ax.spines.values(): spine.set_edgecolor('#2a2e3f')

    plt.tight_layout()
    plt.savefig('01_gradient_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 01_gradient_comparison.png")
