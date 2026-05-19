"""
Train VAE on MNIST
==================
Full training loop with logging, checkpointing, and visualization.

Run:
    python train.py
    python train.py --epochs 20 --latent_dim 10 --beta 4.0
"""

import argparse
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import VAE, elbo_loss


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='Train a VAE on MNIST')
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--latent_dim',  type=int,   default=20)
    p.add_argument('--hidden_dim',  type=int,   default=400)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--beta',        type=float, default=1.0,
                   help='KL weight (1=VAE, >1=β-VAE)')
    p.add_argument('--save_dir',    type=str,   default='outputs')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def get_data(batch_size: int):
    tf = transforms.ToTensor()  # scales [0,255] → [0,1]
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        out = model(x)
        losses = elbo_loss(x, out['x_hat'], out['mu'], out['log_var'], beta=beta)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss  += losses['loss'].item()
        total_recon += losses['recon']
        total_kl    += losses['kl']

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def eval_epoch(model, loader, device, beta):
    model.eval()
    total_loss = total_recon = total_kl = 0.0

    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        losses = elbo_loss(x, out['x_hat'], out['mu'], out['log_var'], beta=beta)
        total_loss  += losses['loss'].item()
        total_recon += losses['recon']
        total_kl    += losses['kl']

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Latent dim: {args.latent_dim} | β: {args.beta} | LR: {args.lr}\n")

    train_loader, test_loader = get_data(args.batch_size)

    model = VAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'kl': [], 'recon': []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_recon, tr_kl = train_epoch(model, train_loader, optimizer, device, args.beta)
        va_loss, va_recon, va_kl = eval_epoch(model,  test_loader,  device, args.beta)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['kl'].append(tr_kl)
        history['recon'].append(tr_recon)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train {tr_loss:.2f} (recon={tr_recon:.2f}, kl={tr_kl:.2f}) | "
              f"val {va_loss:.2f}")

        # Save reconstructions every 2 epochs
        if epoch % 2 == 0:
            x_fixed = next(iter(test_loader))[0][:16].to(device)
            recons  = model.reconstruct(x_fixed)
            grid = torch.cat([x_fixed, recons], dim=0)
            save_image(grid, f"{args.save_dir}/recon_ep{epoch:03d}.png", nrow=16, normalize=True)

            # Save random samples
            samples = model.sample(64, device=device)
            save_image(samples, f"{args.save_dir}/samples_ep{epoch:03d}.png", nrow=8, normalize=True)

    # Save checkpoint
    torch.save({'model_state': model.state_dict(), 'args': vars(args)},
               f"{args.save_dir}/vae_final.pt")
    print(f"\nModel saved to {args.save_dir}/vae_final.pt")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#0b0c10')
    fig.suptitle('VAE Training Curves', color='#e0e4f0', fontsize=13)
    epochs = range(1, args.epochs + 1)

    for ax, key, col, title in zip(
        axes,
        ['train_loss', 'recon', 'kl'],
        ['#5d9cf5',    '#f5a55d', '#5df5a5'],
        ['Total ELBO', 'Reconstruction', 'KL Divergence']
    ):
        ax.set_facecolor('#13151b')
        ax.plot(epochs, history[key], color=col, lw=2)
        ax.set_title(title, color='#e0e4f0')
        ax.set_xlabel('Epoch', color='#717a9e')
        ax.tick_params(colors='#717a9e')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2e3f')

    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
