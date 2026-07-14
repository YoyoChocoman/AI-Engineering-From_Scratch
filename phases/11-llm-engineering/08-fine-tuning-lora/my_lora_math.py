import torch
import torch.nn as nn
import math

# Step 1: Build LoRA external module
class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.linear = original_linear

        for param in self.linear.parameters():
            param.requires_grad = False

        d_in = original_linear.in_features
        d_out = original_linear.out_features
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.rand(d_in, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, d_out))

    def forward(self, x):
        original_output = self.linear(x)
        lora_output = (x @ self.A @ self.B) * self.scaling
        return original_output + lora_output

# Step 2: Test
d_model = 4096
original_layer = nn. Linear(d_model, d_model)

print("The Original Linear Layer:")
total_params = sum(p.numel() for p in original_layer.parameters() if p.requires_grad)
print(f"-> The total number of param participating in the train: {total_params:,}")

print("\nWith LoRA Module:")
lora_layer = LinearWithLoRA(original_layer, rank=16)
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
print(f"-> The forzen params: {frozen_params:,}")
print(f"-> The trainable params: {trainable_params:,}")
print(f"-> Compress propotion: {trainable_params / (trainable_params + frozen_params) * 100:.3f}%")

# Forward Pass
dummy_input = torch.randn(1, 10, d_model)

print("\n=== Start Calculating ===")
output = lora_layer(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")