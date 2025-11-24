"""
Conditional Variational Autoencoder (cVAE) model for high-resolution precipitation generation.

Architecture:
- EncoderX: Encodes low-res multi-variable inputs X_lr (C, 13, 11) → h_X
- EncoderY: Encodes high-res precipitation Y_hr (1, H, W) + statics → h_Y
- LatentHead: Maps [h_Y; h_X] → (mu, logvar) for latent z
- Decoder: Decodes [z; h_X] → Y_hat (1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EncoderX(nn.Module):
    """
    Encoder for low-resolution multi-variable inputs.

    Input: X_lr of shape (B, C, 13, 11)
    Output: h_X of shape (B, d_x)
    """
    
    def __init__(self, in_channels: int = 7, d_x: int = 64, base_filters: int = 32):
        """
        Args:
            in_channels: Number of input channels (variables)
            d_x: Dimension of output embedding
            base_filters: Number of filters in first conv layer
        """
        super().__init__()
    
        self.in_channels = in_channels
        self.d_x = d_x
    
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
    
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
    
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)
    
        # Dynamically calculate flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 13, 11)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.conv3(dummy_output)
            self.flatten_size = dummy_output.view(1, -1).shape[1]
            print(f"[EncoderX] Dynamically computed flatten_size: {self.flatten_size}")
    
        # FC layers to embedding
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, d_x)
    
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, 13, 11)

        Returns:
            h_X: (B, d_x)
        """
        # Conv blocks
        h = self.relu(self.bn1(self.conv1(x)))          # (B, 32, 13, 11)
        h = self.relu(self.bn2(self.conv2(h)))          # (B, 64, 6, 5)
        h = self.relu(self.bn3(self.conv3(h)))          # (B, 128, 3, 2)

        # Flatten
        h = h.flatten(start_dim=1)                      # (B, 768)

        # FC layers
        h = self.relu(self.fc1(h))                      # (B, 256)
        h = self.dropout(h)
        h_X = self.fc2(h)                                # (B, d_x)

        return h_X


class EncoderY(nn.Module):
    """
    Encoder for high-resolution precipitation + static maps.

    Input: Y_hr (B, 1, H, W) concatenated with statics S (B, S_ch, H, W)
    Output: h_Y of shape (B, d_y)
    """

    def __init__(self, in_channels: int = 1, static_channels: int = 2,
                 d_y: int = 256, base_filters: int = 64):
        """
        Args:
            in_channels: Number of channels in Y_hr (usually 1 for precipitation)
            static_channels: Number of static maps
            d_y: Dimension of output embedding
            base_filters: Number of filters in first conv layer
        """
        super().__init__()

        self.in_channels = in_channels
        self.static_channels = static_channels
        self.d_y = d_y

        total_in_ch = in_channels + static_channels

        # Encoder with downsampling
        # Assuming H=156, W=132
        # (B, 1+S_ch, 156, 132) → (B, 64, 156, 132)
        self.conv1 = nn.Conv2d(total_in_ch, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)

        # (B, 64, 156, 132) → (B, 128, 78, 66)
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)

        # (B, 128, 78, 66) → (B, 256, 39, 33)
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)

        # (B, 256, 39, 33) → (B, 512, 19, 16)
        self.conv4 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 8)

        # Calculate flatten size (approximate for H=156, W=132)
        # After 3 stride-2 ops: H/8 ≈ 19, W/8 ≈ 16
        # 512 * 19 * 16 ≈ 155,648
        # We'll use adaptive pooling to ensure consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_size = base_filters * 8 * 8 * 8  # 512 * 64 = 32,768

        # FC layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, d_y)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, y, s):
        """
        Args:
            y: (B, 1, H, W) high-res precipitation
            s: (B, S_ch, H, W) static maps

        Returns:
            h_Y: (B, d_y)
        """
        # Concatenate Y and statics
        x = torch.cat([y, s], dim=1)  # (B, 1+S_ch, H, W)

        # Conv blocks with downsampling
        h = self.relu(self.bn1(self.conv1(x)))   # (B, 64, H, W)
        h = self.relu(self.bn2(self.conv2(h)))   # (B, 128, H/2, W/2)
        h = self.relu(self.bn3(self.conv3(h)))   # (B, 256, H/4, W/4)
        h = self.relu(self.bn4(self.conv4(h)))   # (B, 512, H/8, W/8)

        # Adaptive pooling to ensure consistent size
        h = self.adaptive_pool(h)                # (B, 512, 8, 8)

        # Flatten
        h = h.flatten(start_dim=1)               # (B, 32768)

        # FC layers
        h = self.relu(self.fc1(h))               # (B, 512)
        h = self.dropout(h)
        h_Y = self.fc2(h)                         # (B, d_y)

        return h_Y


class LatentHead(nn.Module):
    """
    Maps concatenated embeddings [h_Y; h_X] to latent distribution parameters.

    Input: [h_Y; h_X] of shape (B, d_y + d_x)
    Output: mu, logvar of shape (B, d_z)
    """

    def __init__(self, d_y: int = 256, d_x: int = 64, d_z: int = 64):
        """
        Args:
            d_y: Dimension of h_Y
            d_x: Dimension of h_X
            d_z: Dimension of latent z
        """
        super().__init__()

        self.d_y = d_y
        self.d_x = d_x
        self.d_z = d_z

        input_dim = d_y + d_x

        self.fc_shared = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, d_z)
        self.fc_logvar = nn.Linear(256, d_z)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, h_Y, h_X):
        """
        Args:
            h_Y: (B, d_y)
            h_X: (B, d_x)

        Returns:
            mu: (B, d_z)
            logvar: (B, d_z)
        """
        h = torch.cat([h_Y, h_X], dim=1)  # (B, d_y + d_x)
        h = self.relu(self.fc_shared(h))  # (B, 256)

        mu = self.fc_mu(h)                 # (B, d_z)
        logvar = self.fc_logvar(h)         # (B, d_z)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder with Transposed Convolutions AND Conditioning Dropout.

    Key features:
    1. Transposed convolutions for learnable upsampling (sharp details)
    2. Conditioning dropout forces the model to use latent variable z
       - During training, h_X is randomly zeroed 50% of the time
       - This prevents posterior collapse by making z essential for reconstruction

    Input: z (B, d_z), h_X (B, d_x)
    Output: Y_hat (B, 1, H, W)
    """

    def __init__(self, d_z: int = 64, d_x: int = 64, H: int = 156, W: int = 132,
                 base_filters: int = 64):
        """
        Args:
            d_z: Dimension of latent z
            d_x: Dimension of h_X conditioning
            H: Target height
            W: Target width
            base_filters: Number of base filters for conv layers
        """
        super().__init__()

        self.d_z = d_z
        self.d_x = d_x
        self.H = H
        self.W = W

        # Starting small feature map dimensions
        self.H_init = 20  # 156 // 8 approx
        self.W_init = 17  # 132 // 8 approx

        self.init_channels = base_filters * 8  # 512

        # CRITICAL: Conditioning Dropout
        # Randomly zeros h_X during training to force model to use z
        # p=0.5 means 50% of batches have h_X zeroed out
        self.cond_dropout = nn.Dropout(p=0.5)

        # FC from [z; h_X] to initial feature map
        input_dim = d_z + d_x
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, self.init_channels * self.H_init * self.W_init),
            nn.ReLU(True)
        )

        # Transposed Convolutions (Learnable upsampling for sharp details)
        # Block 1: (B, 512, 20, 17) → (B, 256, 40, 34) → crop/adjust to (39, 33)
        self.up1 = nn.ConvTranspose2d(
            self.init_channels, base_filters * 4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(base_filters * 4)

        # Block 2: (B, 256, 39, 33) → (B, 128, 78, 66)
        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_filters * 2)

        # Block 3: (B, 128, 78, 66) → (B, 64, 156, 132)
        self.up3 = nn.ConvTranspose2d(
            base_filters * 2, base_filters,
            kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(base_filters)

        # Final output layer
        self.conv_out = nn.Conv2d(base_filters, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, z, h_X):
        """
        Args:
            z: (B, d_z) latent code
            h_X: (B, d_x) conditioning embedding

        Returns:
            Y_hat: (B, 1, H, W) generated precipitation
        """
        # CRITICAL: Apply conditioning dropout during training
        # This forces the decoder to rely on z instead of just h_X
        # During eval/sampling, dropout is disabled automatically
        h_X_dropped = self.cond_dropout(h_X)

        # Concatenate and project to initial feature map
        h = torch.cat([z, h_X_dropped], dim=1)  # (B, d_z + d_x)
        h = self.fc(h)  # (B, init_channels * H_init * W_init)
        h = h.view(-1, self.init_channels, self.H_init, self.W_init)  # (B, 512, 20, 17)

        # Upsample Block 1: (20, 17) → (40, 34) → adjust to (39, 33)
        h = self.relu(self.bn1(self.up1(h)))
        # Handle size mismatch from transposed conv (crop to exact size)
        if h.shape[2] != 39 or h.shape[3] != 33:
            h = F.interpolate(h, size=(39, 33), mode='bilinear', align_corners=False)

        # Upsample Block 2: (39, 33) → (78, 66)
        h = self.relu(self.bn2(self.up2(h)))

        # Upsample Block 3: (78, 66) → (156, 132)
        h = self.relu(self.bn3(self.up3(h)))

        # Final output
        Y_hat = self.conv_out(h)  # (B, 1, 156, 132)

        # Apply ReLU to ensure non-negative precipitation
        Y_hat = F.relu(Y_hat)

        return Y_hat


class CVAE(nn.Module):
    """
    Complete conditional Variational Autoencoder.

    Combines EncoderX, EncoderY, LatentHead, and Decoder.
    """

    def __init__(self,
                 in_channels_X: int = 7,
                 in_channels_Y: int = 1,
                 static_channels: int = 2,
                 d_x: int = 64,
                 d_y: int = 256,
                 d_z: int = 64,
                 H: int = 156,
                 W: int = 132,
                 base_filters: int = 64):
        """
        Args:
            in_channels_X: Number of channels in X_lr
            in_channels_Y: Number of channels in Y_hr (usually 1)
            static_channels: Number of static maps
            d_x: Dimension of h_X embedding
            d_y: Dimension of h_Y embedding
            d_z: Dimension of latent z
            H: Target high-res height
            W: Target high-res width
            base_filters: Base number of convolutional filters
        """
        super().__init__()

        self.encoder_X = EncoderX(in_channels_X, d_x, base_filters=32)
        self.encoder_Y = EncoderY(in_channels_Y, static_channels, d_y, base_filters)
        self.latent_head = LatentHead(d_y, d_x, d_z)
        self.decoder = Decoder(d_z, d_x, H, W, base_filters)

        self.d_z = d_z

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * sigma

        Args:
            mu: (B, d_z) mean
            logvar: (B, d_z) log variance

        Returns:
            z: (B, d_z) sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, X_lr, Y_hr, S):
        """
        Encode inputs to latent distribution parameters.

        Args:
            X_lr: (B, C, H_lr, W_lr)
            Y_hr: (B, 1, H, W)
            S: (B, S_ch, H, W)

        Returns:
            mu: (B, d_z)
            logvar: (B, d_z)
            h_X: (B, d_x)
        """
        h_X = self.encoder_X(X_lr)
        h_Y = self.encoder_Y(Y_hr, S)
        mu, logvar = self.latent_head(h_Y, h_X)
        return mu, logvar, h_X

    def decode(self, z, h_X):
        """
        Decode latent z conditioned on h_X.

        Args:
            z: (B, d_z)
            h_X: (B, d_x)

        Returns:
            Y_hat: (B, 1, H, W)
        """
        return self.decoder(z, h_X)

    def forward(self, X_lr, Y_hr, S):
        """
        Full forward pass through cVAE.

        Args:
            X_lr: (B, C, H_lr, W_lr) low-res inputs
            Y_hr: (B, 1, H, W) high-res precipitation
            S: (B, S_ch, H, W) static maps

        Returns:
            Y_hat: (B, 1, H, W) reconstructed precipitation
            mu: (B, d_z) latent mean
            logvar: (B, d_z) latent log variance
        """
        # Encode
        mu, logvar, h_X = self.encode(X_lr, Y_hr, S)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        Y_hat = self.decode(z, h_X)

        return Y_hat, mu, logvar

    def sample(self, X_lr, S, z=None, use_prior=False):
        """
        Generate sample conditioned on X_lr.

        Args:
            X_lr: (B, C, H_lr, W_lr)
            S: (B, S_ch, H, W)
            z: (B, d_z) optional latent code; if None, sample from prior N(0,I)
            use_prior: If True, sample z ~ N(0,I); otherwise use provided z

        Returns:
            Y_hat: (B, 1, H, W)
        """
        h_X = self.encoder_X(X_lr)

        if z is None or use_prior:
            # Sample from prior
            batch_size = X_lr.shape[0]
            device = X_lr.device
            z = torch.randn(batch_size, self.d_z, device=device)

        Y_hat = self.decode(z, h_X)
        return Y_hat
