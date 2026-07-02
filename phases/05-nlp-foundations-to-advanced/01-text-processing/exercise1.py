import re

def tokenize(text):
    pattern = r"https?://\S+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]"
    return re.findall(pattern, text)

print("=== Exercise 1 ===")
test_url_text = "Visit https://example.com today for 50% off!"
tokens_url = tokenize(test_url_text)
print(f"Original: {test_url_text}")
print(f"After process: {tokens_url}")