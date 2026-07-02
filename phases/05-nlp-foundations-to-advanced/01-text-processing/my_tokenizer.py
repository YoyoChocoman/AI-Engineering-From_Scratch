import re

# Step 1: a regex word tokenizer
def tokenize(text):
    # use regex to seperate words, numbers, and punctuations
    # Rule 1: fetch the words
    # Rule 2: fetch the continuous numbers
    # Rule 3: fetch singular punctuations
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)

print("=== Tokenization ===")
text = "The cats weren't running at 3pm."
tokens = tokenize(text)
print(f"Original: {text}")
print(f"After process: {tokens}\n")

# Step 2: a Porter stemmer (step 1a only)
def stem_step_1a(word):
    if word.endswith("sses"): return word[:-2]
    if word.endswith("ies"): return word[:-2]
    if word.endswith("ss"): return word # ss 不砍 (如 kiss)
    if word.endswith("s") and len(word) > 1: return word[:-1]
    return word

print("=== Stemming ===")
test_words = ["caresses", "ponies", "caress", "cats"]
stems = [stem_step_1a(w) for w in test_words]
print(f"Original: {test_words}")
print(f"After process: {stems}\n")

# Step 3: a lookup-based lemmatizer
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)

    if key in LEMMA_TABLE: return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"): return word[:-3]
    if pos == "NOUN" and word.endswith("s"): return word[:-1]
    return word.lower()

print("=== Lemmatization ===")
print(f"running (VERB) -> {lemmatize('running', 'VERB')}")
print(f"were (VERB)) -> {lemmatize('were', 'VERB')}")
print(f"better (ADJ) -> {lemmatize('better', 'ADJ')}")