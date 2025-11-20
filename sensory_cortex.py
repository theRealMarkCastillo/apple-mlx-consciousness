"""
Sensory Cortex Module
---------------------
Implements the "Visual Cortex" and Multi-Modal Fusion.
Uses raw MLX Convolutional Neural Networks (CNNs) to process images.

Key Concepts:
1. Visual Cortex:
   - A lightweight CNN optimized for the Apple Neural Engine (ANE).
   - Extracts high-level features (shapes, textures) from raw pixels.
   - Output: A dense vector compatible with the Global Workspace.

2. Multi-Modal Fusion:
   - The "Binding Problem" solver.
   - Combines Visual Vectors and Text SDRs into a single "Conscious State".
"""

import mlx.core as mx
import mlx.nn as nn

class VisualCortex(nn.Module):
    """
    A lightweight CNN to encode images into the Global Workspace format.
    Designed to run efficiently on Apple Silicon.
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        
        # Layer 1: Edges & Simple Shapes
        # Input: [Batch, 64, 64, 3] -> Output: [Batch, 32, 32, 16]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        
        # Layer 2: Textures & Parts
        # Input: [Batch, 32, 32, 16] -> Output: [Batch, 16, 16, 32]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # Layer 3: Objects & Concepts
        # Input: [Batch, 16, 16, 32] -> Output: [Batch, 8, 8, 64]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        # Flatten and project to Global Workspace dimension
        # 8 * 8 * 64 = 4096
        self.projection = nn.Linear(4096, output_dim)
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: Image -> Concept Vector.
        Expects x to be [Batch, H, W, 3] or [H, W, 3].
        """
        # Ensure batch dimension
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=0)
            
        # CNN Backbone
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        
        # Flatten
        B = x.shape[0]
        x = x.reshape(B, -1)
        
        # Project to Workspace
        x = self.projection(x)
        
        # Normalize (Brain states are often normalized)
        x = x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        
        return x.squeeze()

class MultiModalFuser(nn.Module):
    """
    Fuses different sensory modalities into a single Global Workspace state.
    Solves the "Binding Problem" via projection and summation.
    """
    def __init__(self, text_dim: int, vision_dim: int, workspace_dim: int):
        super().__init__()
        
        # Project Text SDR to Workspace
        self.text_proj = nn.Linear(text_dim, workspace_dim)
        
        # Project Vision Vector to Workspace (usually already matched, but good for alignment)
        self.vision_proj = nn.Linear(vision_dim, workspace_dim)
        
        # Attention weights (learnable)
        # The brain can focus more on vision or text depending on context
        self.attention = nn.Linear(workspace_dim, 2) # [Text_Weight, Vision_Weight]

    def __call__(self, text_sdr: mx.array, vision_vec: mx.array) -> mx.array:
        """
        Fuse modalities.
        """
        # 1. Project to common space
        t_emb = self.text_proj(text_sdr)
        v_emb = self.vision_proj(vision_vec)
        
        # 2. Simple Fusion (Summation)
        # In GWT, modalities compete/cooperate to enter the workspace.
        # Here we sum them, allowing both to influence the state.
        fused_state = t_emb + v_emb
        
        # 3. Normalize
        fused_state = fused_state / (mx.linalg.norm(fused_state) + 1e-6)
        
        return fused_state
