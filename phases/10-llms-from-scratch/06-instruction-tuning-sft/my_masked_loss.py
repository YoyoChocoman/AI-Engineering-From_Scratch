import torch
import torch.nn.functional as F

print("=== SFT's Core Secret: Masked Loss ===")

# Step 1: Prepare a conversation
prompt_tokens = [10, 20, 30]
answer_tokens = [40, 50, 60]

full_sequence = torch.tensor([prompt_tokens + answer_tokens])
print(f"Full conversation tokens: {full_sequence.tolist()[0]}")

# Step 2: Loss Mask
mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
print(f"Mask: {mask.tolist()[0]}")

# Step 3: Simulate GPT's prediction (Logits)
torch.manual_seed(42)
logits = torch.randn(1, 6, 1000)

# Step 4: Targets
targets = full_sequence.clone()

# Test
loss_unmasked = F.cross_entropy(logits.view(-1, 1000), targets.view(-1), reduction='none')
bad_final_loss = loss_unmasked.mean()

print(f"\nWithout Mask: {bad_final_loss.item():.4f}")

loss_masked = loss_unmasked * mask.view(-1)

num_response_tokens = mask.sum()
good_final_loss = loss_masked.sum() / num_response_tokens

print(f"\nWith Mask: {good_final_loss.item():.4f}")