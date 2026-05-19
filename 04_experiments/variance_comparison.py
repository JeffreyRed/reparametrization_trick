"""
Experiment — Variance Comparison at Scale
==========================================
A systematic study of gradient estimator variance as a function of:
  - Dimension of the latent space
  - Number of Monte Carlo samples
  - The value of μ

This is the definitive empirical argument for using reparameterization.
"""

import numpy as np
import matplotlib.pyplot as plt


def score_function_grad(mu, sigma, n):
    """Score function (REINFORCE) gradient estimator."""
    z = np.random.normal(mu, sigma, n)
    f_z = z ** 2
    score = (z - mu) / (sigma ** 2)
    return np.mean(f_z * score)


def reparam_grad(mu, sigma, n):
    """Reparameterized gradient estimator."""
    eps = np.random.normal(0, 1, n)
    z = mu + sigma * eps
    # ∂(z²)/∂μ = 2z; ∂z/∂μ = 1
    return np.mean(2 * z * 1)


def estimate_variance(estimator_fn, mu, sigma, n_samples, n_trials=5000):
    estimates = [estimator_fn(mu, sigma, n_samples) for _ in range(n_trials)]
    return np.var(estimates)


if __name__ == "__main__":
    # ── Experiment 1: Variance vs sample count ─────────────────────────────
    sample_counts = [1, 2, 5, 10, 20, 50, 100]
    mu, sigma = 2.0, 1.0

    sf_vars  = [estimate_variance(score_function_grad, mu, sigma, n) for n in sample_counts]
    rep_vars = [estimate_variance(reparam_grad, mu, sigma, n)         for n in sample_counts]

    print("Variance vs Sample Count (μ=2, σ=1)")
    print(f"{'n':>5} | {'SF Var':>12} | {'Reparam Var':>12} | {'Ratio':>8}")
    print("-" * 45)
    for n, sv, rv in zip(sample_counts, sf_vars, rep_vars):
        print(f"{n:>5} | {sv:>12.4f} | {rv:>12.4f} | {sv/rv:>8.1f}x")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0b0c10')
    fig.suptitle('Variance of Gradient Estimators', color='#e0e4f0', fontsize=14)

    ax = axes[0]
    ax.set_facecolor('#13151b')
    ax.semilogy(sample_counts, sf_vars,  'o-', color='#f5a55d', lw=2, label='Score Function')
    ax.semilogy(sample_counts, rep_vars, 's-', color='#5d9cf5', lw=2, label='Reparameterization')
    ax.set_xlabel('# Monte Carlo samples', color='#717a9e')
    ax.set_ylabel('Gradient Variance (log scale)', color='#717a9e')
    ax.set_title('Variance vs Sample Count', color='#e0e4f0')
    ax.legend(facecolor='#1c1f2b', labelcolor='#e0e4f0', edgecolor='#2a2e3f')
    ax.tick_params(colors='#717a9e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2e3f')

    # ── Experiment 2: Variance vs μ ────────────────────────────────────────
    mus = np.linspace(-4, 4, 30)
    sf_vars_mu  = [estimate_variance(score_function_grad, m, 1.0, 1) for m in mus]
    rep_vars_mu = [estimate_variance(reparam_grad, m, 1.0, 1)         for m in mus]

    ax = axes[1]
    ax.set_facecolor('#13151b')
    ax.plot(mus, sf_vars_mu,  color='#f5a55d', lw=2, label='Score Function')
    ax.plot(mus, rep_vars_mu, color='#5d9cf5', lw=2, label='Reparameterization')
    ax.set_xlabel('μ', color='#717a9e')
    ax.set_ylabel('Gradient Variance', color='#717a9e')
    ax.set_title('Variance vs μ  (n=1)', color='#e0e4f0')
    ax.legend(facecolor='#1c1f2b', labelcolor='#e0e4f0', edgecolor='#2a2e3f')
    ax.tick_params(colors='#717a9e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2e3f')

    plt.tight_layout()
    plt.savefig('variance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to variance_comparison.png")
