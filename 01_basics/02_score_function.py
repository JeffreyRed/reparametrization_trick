"""
Exercise 02 — Score Function (REINFORCE) Estimator
====================================================
Goal: Understand the alternative gradient estimator and see why
      reparameterization is preferred (lower variance).

The score function estimator (a.k.a. REINFORCE, log-derivative trick):
    ∇_φ 𝔼_{z~q_φ}[f(z)]  =  𝔼_{z~q_φ}[ f(z) · ∇_φ log q_φ(z) ]

This is unbiased but has HIGH variance. We'll compare it to the
reparameterization estimator on the same objective.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Objective: f(μ) = 𝔼_{z ~ 𝒩(μ, 1)}[ z² ]
# True gradient: ∂f/∂μ = 2μ
# ──────────────────────────────────────────────────────────────────────────────

def score_function_estimator(mu_val: float, n_samples: int = 1) -> float:
    """
    Score function estimator:
        ∇μ 𝔼[f(z)] = 𝔼[ f(z) · ∇μ log q(z;μ) ]
    
    For z ~ 𝒩(μ, 1):
        log q(z;μ) = -0.5*(z-μ)² + const
        ∇μ log q(z;μ) = (z - μ)      ← the "score"
    
    So the estimator is:
        ∇μ f ≈ (1/N) Σ f(z_i) · (z_i - μ)
    """
    mu = mu_val
    z_samples = np.random.normal(mu, 1.0, n_samples)
    f_z = z_samples ** 2                        # objective values
    score = z_samples - mu                      # ∇μ log q(z;μ)
    gradient_estimate = np.mean(f_z * score)    # estimator
    return gradient_estimate


def reparam_estimator(mu_val: float, n_samples: int = 1) -> float:
    """
    Reparameterization estimator:
        z = μ + ε,  ε ~ 𝒩(0,1)
        ∇μ f(z) = ∂f/∂z · ∂z/∂μ = 2z · 1 = 2z
    """
    mu = mu_val
    eps = np.random.normal(0, 1, n_samples)
    z = mu + eps
    gradient_estimate = np.mean(2 * z)          # ∂(z²)/∂μ = 2z
    return gradient_estimate


# ──────────────────────────────────────────────────────────────────────────────
# Variance comparison experiment
# ──────────────────────────────────────────────────────────────────────────────
def compare_variance(mu: float, n_trials: int = 5000, n_samples_per_trial: int = 1):
    """Run many trials and measure the variance of each estimator."""
    true_grad = 2 * mu

    sf_estimates   = [score_function_estimator(mu, n_samples_per_trial) for _ in range(n_trials)]
    rep_estimates  = [reparam_estimator(mu, n_samples_per_trial) for _ in range(n_trials)]

    sf_var  = np.var(sf_estimates)
    rep_var = np.var(rep_estimates)

    print(f"μ = {mu}")
    print(f"  True gradient:              {true_grad:.4f}")
    print(f"  Score Function — mean:      {np.mean(sf_estimates):.4f}  |  var: {sf_var:.4f}")
    print(f"  Reparameterization — mean:  {np.mean(rep_estimates):.4f}  |  var: {rep_var:.4f}")
    print(f"  Variance ratio (SF/Rep):    {sf_var / (rep_var + 1e-12):.1f}x higher\n")

    return sf_estimates, rep_estimates


if __name__ == "__main__":
    print("=" * 55)
    print("Gradient Estimator Variance Comparison")
    print("=" * 55)

    mu_test = 2.0
    sf_ests, rep_ests = compare_variance(mu=mu_test, n_trials=5000)

    # ── Plot distributions of gradient estimates ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0b0c10')
    fig.suptitle(f'Gradient Estimators at μ={mu_test}  (true grad = {2*mu_test})',
                 color='#e0e4f0', fontsize=13)

    true_grad = 2 * mu_test

    for ax, data, title, col in zip(
        axes,
        [sf_ests, rep_ests],
        ['Score Function (REINFORCE)', 'Reparameterization'],
        ['#f5a55d', '#5d9cf5']
    ):
        ax.set_facecolor('#13151b')
        clip = np.percentile(np.abs(data), 98)
        data_clipped = np.clip(data, -clip, clip)
        ax.hist(data_clipped, bins=80, color=col, alpha=0.75, edgecolor='none')
        ax.axvline(true_grad, color='#5df5a5', lw=2, label=f'True grad = {true_grad}')
        ax.set_title(title, color='#e0e4f0', pad=8)
        ax.set_xlabel('Gradient estimate', color='#717a9e')
        ax.set_ylabel('Count', color='#717a9e')
        ax.tick_params(colors='#717a9e')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2e3f')
        ax.legend(facecolor='#1c1f2b', labelcolor='#e0e4f0', edgecolor='#2a2e3f')
        ax.text(0.98, 0.95, f'Var = {np.var(data):.2f}',
                transform=ax.transAxes, ha='right', va='top',
                color=col, fontsize=11, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('02_variance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 02_variance_comparison.png")

    # ── Exercise: Try different sample counts ──
    print("\n--- Effect of sample size (n_samples) on variance ---")
    for n in [1, 5, 20, 100]:
        sf_v  = np.var([score_function_estimator(mu_test, n) for _ in range(3000)])
        rep_v = np.var([reparam_estimator(mu_test, n) for _ in range(3000)])
        print(f"  n={n:3d}:  SF var={sf_v:8.3f}  |  Reparam var={rep_v:8.3f}  |  ratio={sf_v/rep_v:.1f}x")
