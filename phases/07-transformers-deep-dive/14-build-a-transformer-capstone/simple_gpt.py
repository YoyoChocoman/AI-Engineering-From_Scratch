import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()

        # 1. Dictionary (Embedding)：turn words into matrix
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # 2. Navigate (Positional)：add position info(use a simplified version to replace RoPE)
        self.pos_emb = nn.Embedding(1024, d_model)

        # 3. Engine (Blocks)：repeat Transformer Block n_layers times！
        # (including Attention, FFN, residual conneciton, mask)
        self.blocks = nn.Sequential(
            *[Block(d_model) for _ in range(n_layers)]
        )

        # 4. normalization (RMSNorm)
        self.ln_f = nn.LayerNorm(d_model)

        # 5. LM Head：turn matrix into prob
        self.lm_head = nn.Linear(d_model, vocab_size)

    # Forwarding (Forward Pass)
    def forward(self, idx):
        B, T = idx.shape # B: batch size, T: sentense length

        # Step A: turn words into vector with position info
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))

        # Step B: flow throgh all transformer block
        x = self.blocks(x)

        # Step C: predict the next word prob (logits)
        logits = self.lm_head(self.ln_f(x))

        return logits

# then the logits will go thru softmax turns into prob
# and thru temp sampling the gpt will split ou the next word