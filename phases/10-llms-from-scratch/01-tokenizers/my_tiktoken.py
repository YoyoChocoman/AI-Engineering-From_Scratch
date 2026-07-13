# ive already written the BPE in phase 5
# so i skip the BPE part
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

print("=== Mutilingual Tax ===")

# same sentence in different language
texts = {
    "English": "Natural language processing is the study of how computers understand human language.",
    "Spanish": "El procesamiento del lenguaje natural es el estudio de cómo las computadoras entienden el lenguaje humano.",
    "Chinese": "自然語言處理是研究電腦如何理解人類語言的學科。",
    "Korean": "자연어 처리는 컴퓨터가 인간의 언어를 이해하는 방법을 연구하는 학문입니다."
}

for lang, text in texts.items():
    char_length = len(text)
    tokens = enc.encode(text)
    token_length = len(tokens)

    ratio = token_length / char_length

    print(f"\n[{lang}]")
    print(f"Text: {text}")
    print(f"Length Comparison: {char_length} chars -> {token_length} tokens")
    print(f"Ratio: {ratio:.2f} tokens/char")

    pieces = [enc.decode([t]) for t in tokens[:5]]
    print(f"Pieces: {pieces} ...")