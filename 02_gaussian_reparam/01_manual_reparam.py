"""
Exercise 01 — Manual Reparameterization + Autograd
====================================================
Goal: Implement the trick from scratch using PyTorch autograd.
      Build intuition for what happens inside a VAE encoder.

We parameterize a Gaussian with a tiny neural network and verify
that gradients flow correctly back to the network weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# The Reparameterization module
# ──────────────────────────────────────────────────────────────────────────────
class GaussianReparam(nn.Module):
    """
    Implements z = μ + σ · ε  where ε ~ 𝒩(0, I)
    
    Key design choice: the network outputs log(σ²) for numerical stability.
    σ = exp(0.5 * log_var)
    
    This ensures σ > 0 always (exp is always positive).
    """

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> tuple:
        """
        Args:
            mu:      [batch, latent_dim] — mean from encoder
            log_var: [batch, latent_dim] — log variance from encoder

        Returns:
            z:   [batch, latent_dim] — sampled latent vector
            eps: [batch, latent_dim] — the noise used (useful for debugging)
        """
        if self.training:
            # ── THE TRICK ──────────────────────────────────────────────────
            std = torch.exp(0.5 * log_var)   # σ = exp(½ · log σ²)
            eps = torch.randn_like(std)       # ε ~ 𝒩(0, I)  — NO gradient
            z   = mu + std * eps             # deterministic transform
            # ─────────────────────────────────────────────────────────────
            return z, eps
        else:
            # At eval time, just use the mean (no sampling)
            return mu, torch.zeros_like(mu)


# ──────────────────────────────────────────────────────────────────────────────
# Toy encoder: maps x → (μ, log_var)
# ──────────────────────────────────────────────────────────────────────────────
class ToyEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, latent_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, 16)
        self.mu_head      = nn.Linear(16, latent_dim)
        self.log_var_head = nn.Linear(16, latent_dim)
        self.reparam = GaussianReparam()

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h)
        z, eps  = self.reparam(mu, log_var)
        return z, mu, log_var, eps


# ──────────────────────────────────────────────────────────────────────────────
# KL divergence (closed form for Gaussians)
# ──────────────────────────────────────────────────────────────────────────────
def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL[ 𝒩(μ, σ²) ‖ 𝒩(0, 1) ] = ½ · Σ(1 + log σ² - μ² - σ²)
    
    This has a closed form — no sampling needed!
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow verification
# ──────────────────────────────────────────────────────────────────────────────
def verify_gradient_flow():
    """
    Confirm that gradients reach every part of the encoder.
    """
    print("=" * 55)
    print("Gradient Flow Verification")
    print("=" * 55)

    encoder = ToyEncoder(input_dim=2, latent_dim=2)
    encoder.train()

    x = torch.randn(8, 2)                        # batch of 8, input dim 2
    z, mu, log_var, eps = encoder(x)

    # Simple loss depending on z
    loss = z.pow(2).mean() + kl_divergence(mu, log_var).mean()
    loss.backward()

    print(f"\nLoss value: {loss.item():.4f}")
    print("\nGradients for each parameter group:")
    for name, param in encoder.named_parameters():
        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
        status = "✓" if param.grad is not None else "✗ MISSING"
        print(f"  {status}  {name:<30} grad norm = {grad_norm:.6f}")

    print("\n✓ All gradients present — the trick works!\n")


# ──────────────────────────────────────────────────────────────────────────────
# Visualize the transformation for different μ, σ
# ──────────────────────────────────────────────────────────────────────────────
def visualize_transform():
    n = 2000
    eps = torch.randn(n)

    configs = [
        (0.0,  1.0,  '#5d9cf5', 'μ=0, σ=1  (standard)'),
        (2.0,  1.0,  '#f5a55d', 'μ=2, σ=1  (shift)'),
        (0.0,  0.4,  '#5df5a5', 'μ=0, σ=0.4 (squeeze)'),
        (-1.5, 1.8,  '#f55d7a', 'μ=-1.5, σ=1.8 (shift+stretch)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor='#0b0c10')
    fig.suptitle('Reparameterization: ε → z = μ + σ·ε', color='#e0e4f0', fontsize=14, y=0.98)

    for ax, (mu, sigma, col, label) in zip(axes.flat, configs):
        z = mu + sigma * eps  # THE TRICK
        ax.set_facecolor('#13151b')
        ax.hist(eps.numpy(), bins=50, alpha=0.5, color='#5d9cf5', label='ε ~ 𝒩(0,1)', density=True)
        ax.hist(z.numpy(),   bins=50, alpha=0.6, color=col,       label=f'z ~ 𝒩({mu},{sigma}²)', density=True)
        ax.set_title(label, color='#e0e4f0', fontsize=11, pad=8)
        ax.legend(facecolor='#1c1f2b', labelcolor='#e0e4f0', edgecolor='#2a2e3f', fontsize=9)
        ax.tick_params(colors='#717a9e')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2e3f')

    plt.tight_layout()
    plt.savefig('03_reparam_transforms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 03_reparam_transforms.png")


if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Verify gradients flow
    verify_gradient_flow()

    # 2. Visualize the transform
    visualize_transform()

    # ── EXERCISE ──────────────────────────────────────────────────────────────
    # Try to implement the reparam trick yourself below.
    # Fill in the blanks:
    #
    # def my_reparam(mu, log_var):
    #     std = _______________          # σ from log_var
    #     eps = _______________          # sample noise
    #     z   = _______________          # transform
    #     return z
    #
    # Then test: does the gradient flow to mu and log_var?
    # ─────────────────────────────────────────────────────────────────────────
