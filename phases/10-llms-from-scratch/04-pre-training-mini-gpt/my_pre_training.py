# i decide to write with oytorch instead since numpy code is outdated

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== PyTorch Mini Training Loop ===")

# A fake GPT model
class TinyFakeGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # pretend it's a 12-layer transformer
        self.fake_transformer = nn.Linear(embed_dim, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fake_transformer(x))
        return self.lm_head(x)

vocab_size = 1000
embed_dim = 64
seq_len = 16
batch_size = 4
epochs = 100

model = TinyFakeGPT(vocab_size, embed_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters())}")
print("Start guessing the next word...")

# Training Loop
fake_data = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
for epoch in range(epochs):
    # Step A: Form random article as Input (the first 15 char)
    #         Ans is the Input shifted by 1 (2 ~ 16 char)
    inputs = fake_data[:, :-1]
    targets = fake_data[:, 1:]

    # Step B: Clear the last loop's result (zero gradint)
    optimizer.zero_grad()

    # Step C: Start training (Forward Pass)
    logits = model(inputs)

    # Step D: Calculate the score (Croass Entropy)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))

    # Step E: Backward Pass
    loss.backward()

    # Step F: Revise the weight
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"{epoch + 1:3d} Round | Loss: {loss.item():.4f}")