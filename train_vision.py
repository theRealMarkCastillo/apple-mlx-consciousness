"""
Train Visual Cortex
-------------------
Trains the Visual Cortex (CNN) to align with the Text SDR space.
This grounds visual perception in language concepts (CLIP-style alignment).

Process:
1. Generate synthetic images (Red Triangle, Blue Square, etc.).
2. Generate corresponding text labels.
3. Encode text labels into the Global Workspace (Target Vectors).
4. Train Visual Cortex to output vectors that match the Target Vectors.
5. Save weights for the Conscious Chatbot.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
from PIL import Image, ImageDraw

from sensory_cortex import VisualCortex, MultiModalFuser
from biological_nlp import SDREncoder

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255)
}

SHAPES = ["square", "circle", "triangle"]

def generate_synthetic_batch(batch_size=32):
    """
    Generates a batch of (Image, Label) pairs.
    """
    images = []
    labels = []
    
    for _ in range(batch_size):
        # Pick random attributes
        color_name = np.random.choice(list(COLORS.keys()))
        shape_name = np.random.choice(SHAPES)
        
        # Create Label
        label = f"{color_name} {shape_name}"
        labels.append(label)
        
        # Create Image
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        draw = ImageDraw.Draw(img)
        
        color_rgb = COLORS[color_name]
        
        # Random position and size
        size = np.random.randint(15, 30)
        x = np.random.randint(0, IMG_SIZE - size)
        y = np.random.randint(0, IMG_SIZE - size)
        
        if shape_name == "square":
            draw.rectangle([x, y, x+size, y+size], fill=color_rgb)
        elif shape_name == "circle":
            draw.ellipse([x, y, x+size, y+size], fill=color_rgb)
        elif shape_name == "triangle":
            points = [
                (x + size//2, y),           # Top
                (x, y + size),              # Bottom Left
                (x + size, y + size)        # Bottom Right
            ]
            draw.polygon(points, fill=color_rgb)
            
        # Convert to MLX array [H, W, 3] normalized
        img_arr = np.array(img, dtype=np.float32) / 255.0
        images.append(img_arr)
        
    return mx.array(np.stack(images)), labels

def train():
    print("🧠 Initializing Models...")
    
    # 1. Text Encoder (Fixed)
    sdr_encoder = SDREncoder()
    
    # 2. Multi-Modal Fuser (To get the Text Projection)
    # We need to save this later so the chatbot uses the same projection
    fuser = MultiModalFuser(text_dim=2048, vision_dim=128, workspace_dim=128)
    
    # 3. Visual Cortex (To be trained)
    visual_cortex = VisualCortex(output_dim=128)
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    
    # Loss Function: Cosine Embedding Loss (maximize similarity)
    def loss_fn(model, images, target_vectors):
        # Forward pass vision
        vision_vectors = model(images)
        
        # Cosine Similarity = (A . B) / (|A| |B|)
        # Vectors are already normalized by the model/fuser, but let's be safe
        vision_norm = vision_vectors / (mx.linalg.norm(vision_vectors, axis=-1, keepdims=True) + 1e-6)
        target_norm = target_vectors / (mx.linalg.norm(target_vectors, axis=-1, keepdims=True) + 1e-6)
        
        similarity = mx.sum(vision_norm * target_norm, axis=-1)
        
        # Loss = 1 - mean_similarity (Minimize distance)
        return 1.0 - mx.mean(similarity)
    
    loss_and_grad = nn.value_and_grad(visual_cortex, loss_fn)
    
    print(f"🚀 Starting Training ({EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        # Generate Data
        images, text_labels = generate_synthetic_batch(BATCH_SIZE)
        
        # Generate Targets (Text Grounding)
        # We project the text SDRs into the Workspace using the Fuser's projection layer
        target_vectors = []
        for text in text_labels:
            sdr = sdr_encoder.encode(text)
            # Project using fuser (we treat fuser weights as fixed ground truth for now)
            vec = fuser.text_proj(sdr)
            target_vectors.append(vec)
        target_vectors = mx.stack(target_vectors)
        
        # Train Step
        loss, grads = loss_and_grad(visual_cortex, images, target_vectors)
        optimizer.update(visual_cortex, grads)
        mx.eval(visual_cortex.parameters(), optimizer.state)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
            
    print("✅ Training Complete.")
    
    # Save Weights
    print("💾 Saving weights...")
    visual_cortex.save_weights("visual_cortex.npz")
    fuser.save_weights("multimodal_fuser.npz")
    
    # Verification
    print("\n🔍 Verifying...")
    test_img, test_labels = generate_synthetic_batch(1)
    print(f"   Input: {test_labels[0]}")
    
    vec = visual_cortex(test_img)
    
    # Check similarity with target
    sdr = sdr_encoder.encode(test_labels[0])
    target = fuser.text_proj(sdr)
    
    sim = mx.sum((vec/mx.linalg.norm(vec)) * (target/mx.linalg.norm(target))).item()
    print(f"   Cosine Similarity to Label: {sim:.4f}")

if __name__ == "__main__":
    train()
