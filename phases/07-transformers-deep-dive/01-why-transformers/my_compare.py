import time
import random

# RNN simulation
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x
    return h

# Transformer (Attention) simulation
def attention_style(xs):
    return sum(xs) / len(xs)

# test
xs = [random.random() for _ in range(10_000_000)]

print("start RNN mode ....")
start = time.time()
rnn_result = rnn_style(xs)
rnn_time = time.time() - start
print(f"RNN spent: {rnn_time:.4f} seconds (result: {rnn_result:.2f})")

print("start Transformer mode ....")
start = time.time()
attn_result = attention_style(xs)
attn_time = time.time() - start
print(f"Transformer spent: {attn_time:.4f} seconds (result: {attn_result:.2f})")

print(f"Transformer is {rnn_time / attn_time} times faster than RNN")