"""
VAE Model — Encoder + Decoder with Reparameterization
=======================================================
A clean, well-commented VAE for MNIST (28×28 grayscale images).

Key design decisions explained inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Maps x → (μ, log_var) of the approximate posterior q_φ(z|x).
    
    We output log_var (not var or std) because:
      - log_var ∈ ℝ (unconstrained), easier to optimize
      - σ = exp(0.5 * log_var) is always positive
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, 784] → mu, log_var: [batch, latent_dim]"""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Maps z → x̂ (reconstruction of the input).
    
    Output uses sigmoid to keep values in [0,1] (pixel values).
    This pairs with binary cross-entropy loss.
    """
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400, output_dim: int = 784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, latent_dim] → x_hat: [batch, 784]"""
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_hat = torch.sigmoid(self.fc3(h))
        return x_hat


# ──────────────────────────────────────────────────────────────────────────────
# VAE: puts it all together
# ──────────────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        THE REPARAMETERIZATION TRICK
        ================================
        During training:
            ε ~ 𝒩(0, I)
            z = μ + σ · ε       where σ = exp(0.5 · log_var)

        During eval:
            z = μ  (deterministic — use the mean)

        Why training-only sampling?
            At test time we often want the most likely reconstruction,
            not a random one.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)     # σ = exp(½ log σ²)
            eps = torch.randn_like(std)         # ε ~ 𝒩(0, I)
            return mu + std * eps              # z = μ + σ·ε  ← in the graph!
        else:
            return mu

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.
        Returns a dict so we can easily log all quantities.
        """
        # Flatten image
        x_flat = x.view(x.size(0), -1)           # [batch, 784]

        # Encode
        mu, log_var = self.encoder(x_flat)        # [batch, latent_dim] each

        # Reparameterize
        z = self.reparameterize(mu, log_var)      # [batch, latent_dim]

        # Decode
        x_hat = self.decoder(z)                   # [batch, 784]

        return {
            'x_hat':   x_hat,
            'mu':      mu,
            'log_var': log_var,
            'z':       z,
        }

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu') -> torch.Tensor:
        """Generate n new samples from the prior p(z) = 𝒩(0, I)."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        x_hat = self.decoder(z)
        return x_hat.view(n, 1, 28, 28)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input through the VAE."""
        self.eval()
        out = self.forward(x)
        return out['x_hat'].view(-1, 1, 28, 28)


# ──────────────────────────────────────────────────────────────────────────────
# ELBO Loss
# ──────────────────────────────────────────────────────────────────────────────
def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor,
              mu: torch.Tensor, log_var: torch.Tensor,
              beta: float = 1.0) -> dict:
    """
    Evidence Lower BOund (ELBO) loss:
        ℒ = 𝔼[log p(x|z)] - β · KL[ q(z|x) ‖ p(z) ]

    Two terms:
        1. Reconstruction: how well does x̂ match x?  (Binary Cross-Entropy)
        2. KL divergence:  how close is q(z|x) to 𝒩(0, I)?  (Closed form!)

    β > 1  → more regularization → smoother latent space (β-VAE)
    β = 1  → standard VAE

    Args:
        x:       original input [batch, 784]
        x_hat:   reconstruction [batch, 784]
        mu:      encoder mean [batch, latent_dim]
        log_var: encoder log variance [batch, latent_dim]
        beta:    KL weight (default 1.0)
    """
    batch_size = x.size(0)

    # Reconstruction loss (sum over pixels, mean over batch)
    recon = F.binary_cross_entropy(x_hat, x.view(batch_size, -1),
                                   reduction='sum') / batch_size

    # KL divergence (closed form for Gaussians)
    # KL[ 𝒩(μ,σ²) ‖ 𝒩(0,1) ] = ½ Σ(1 + log σ² - μ² - σ²)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

    total = recon + beta * kl

    return {'loss': total, 'recon': recon.item(), 'kl': kl.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VAE(latent_dim=20)
    model.train()

    x = torch.randn(16, 1, 28, 28)   # fake batch
    out = model(x)

    losses = elbo_loss(x, out['x_hat'], out['mu'], out['log_var'])
    losses['loss'].backward()

    print("Forward pass OK")
    print(f"  Reconstruction loss: {losses['recon']:.4f}")
    print(f"  KL divergence:       {losses['kl']:.4f}")
    print(f"  Total ELBO loss:     {losses['loss'].item():.4f}")

    # Check gradients reach the encoder
    for name, p in model.encoder.named_parameters():
        assert p.grad is not None, f"Missing grad: {name}"
    print("\n✓ Gradients flow through reparameterization to encoder!")
