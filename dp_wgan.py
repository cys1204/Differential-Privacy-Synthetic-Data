import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import pandas as pd
import numpy as np
import os
import argparse

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_dp_wgan(data_path, output_path, epochs=50, batch_size=128, z_dim=64, epsilon=5.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for DP-WGAN")
    
    # Load and normalize data
    df = pd.read_csv(data_path)
    # We assume 'income' is the last column (index 12)
    feature_cols = df.columns
    data = df.values.astype(np.float32)
    
    # Simple Min-Max scaling to [-1, 1] for stable GAN training
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    # Avoid division by zero
    diff = data_max - data_min
    diff[diff == 0] = 1.0
    data_scaled = 2.0 * (data - data_min) / diff - 1.0
    
    dataset = TensorDataset(torch.from_numpy(data_scaled))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    data_dim = data.shape[1]
    
    generator = Generator(z_dim, data_dim).to(device)
    discriminator = Discriminator(data_dim).to(device)
    
    # Optimizers
    opt_g = optim.RMSprop(generator.parameters(), lr=0.0001)
    opt_d = optim.RMSprop(discriminator.parameters(), lr=0.0001)
    
    # Attach PrivacyEngine only to discriminator
    privacy_engine = PrivacyEngine()
    discriminator, opt_d, dataloader = privacy_engine.make_private_with_epsilon(
        module=discriminator,
        optimizer=opt_d,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )
    
    print(f"Starting DP-WGAN Training (Target Epsilon: {epsilon})...")
    
    # Training Loop
    n_critic = 5
    for epoch in range(epochs):
        for i, (real_data,) in enumerate(dataloader):
            real_data = real_data.to(device)
            cur_batch_size = real_data.shape[0]
            
            # --- Train Discriminator ---
            opt_d.zero_grad()
            
            z = torch.randn(cur_batch_size, z_dim).to(device)
            fake_data = generator(z).detach()
            
            # Combine real and fake to compute per-sample gradients in one go for Opacus
            combined_data = torch.cat([real_data, fake_data], dim=0)
            combined_validity = discriminator(combined_data)
            
            real_validity = combined_validity[:cur_batch_size]
            fake_validity = combined_validity[cur_batch_size:]
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            d_loss.backward()
            opt_d.step()
            
            # Weight clipping for WGAN (Lipschitz constraint)
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # --- Train Generator ---
            if i % n_critic == 0:
                opt_g.zero_grad()
                z = torch.randn(cur_batch_size, z_dim).to(device)
                fake_data = generator(z)
                
                # Temporarily disable discriminator gradients
                for p in discriminator.parameters():
                    p.requires_grad = False
                    
                discriminator.disable_hooks()
                g_loss = -torch.mean(discriminator(fake_data))
                g_loss.backward()
                opt_g.step()
                discriminator.enable_hooks()
                
                for p in discriminator.parameters():
                    p.requires_grad = True
                
        if epoch % 10 == 0 or epoch == epochs - 1:
            epsilon_spent = privacy_engine.get_epsilon(1e-5)
            print(f"[Epoch {epoch}/{epochs}] D loss: {d_loss.item():.4f} G loss: {g_loss.item():.4f} Epsilon: {epsilon_spent:.4f}")

    # Generate synthetic data
    generator.eval()
    num_samples = len(df)
    z = torch.randn(num_samples, z_dim).to(device)
    with torch.no_grad():
        synthetic_scaled = generator(z).cpu().numpy()
        
    # Inverse transform
    synthetic_data = (synthetic_scaled + 1.0) / 2.0 * diff + data_min
    # Round categorical/integer columns (Adult data is mostly integers)
    synthetic_data = np.round(synthetic_data).astype(int)
    
    # Ensure targets are 0 or 1
    target_idx = list(feature_cols).index('income')
    synthetic_data[:, target_idx] = np.clip(synthetic_data[:, target_idx], 0, 1)
    
    synth_df = pd.DataFrame(synthetic_data, columns=feature_cols)
    synth_df.to_csv(output_path, index=False)
    print(f"DP-WGAN Synthetic Data saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='anonymized_data/adult_cleaned.csv')
    parser.add_argument('--output', type=str, default='anonymized_data/adult_dp_wgan.csv')
    parser.add_argument('--epsilon', type=float, default=5.0)
    args = parser.parse_args()
    
    train_dp_wgan(args.input, args.output, epsilon=args.epsilon)
