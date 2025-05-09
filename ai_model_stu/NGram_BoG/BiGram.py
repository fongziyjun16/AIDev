from collections import defaultdict, Counter

corpus = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "她喜欢吃葡萄",
    "他不喜欢吃香蕉",
    "他喜欢吃苹苹果",
    "她喜欢吃草莓",
]

def tokenize(text):
    return [char for char in text]

def count_grams(corpus, n):
    ngrams_count = defaultdict(Counter)
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i: i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            ngrams_count[prefix][token] += 1
    return ngrams_count

bigram_counts = count_grams(corpus, 2)

# for prefix, counts in bigram_counts.items():
#     print(f"{"".join(prefix)}: {dict(counts)}")

def ngram_probabilities(ngram_counts):
    ngram_probs = defaultdict(Counter)
    for prefix, tokens_count in ngram_counts.items():
        total_count = sum(tokens_count.values())
        for token, count in tokens_count.items():
            ngram_probs[prefix][token] = count / total_count
    return ngram_probs

bigram_probs = ngram_probabilities(bigram_counts)

# for prefix, probs in bigram_probs.items():
#     print(f"{"".join(prefix)}: {dict(probs)}")

def generate_next_token(prefix, ngram_probs):
    if not prefix in bigram_probs:
        return None
    next_token_probs = ngram_probs[prefix]
    next_token = max(next_token_probs, key=next_token_probs.get)
    return next_token

def generate_text(prefix, ngram_probs, n, length=6):
    tokens = tokenize(prefix)
    for _ in range(length - len(prefix)):
        next_token = generate_next_token(tuple(tokens[-(n-1):]), ngram_probs)
        if not next_token:
            break
        tokens.append(next_token)
    return "".join(tokens)

generated_text = generate_text("不喜欢", bigram_probs, 2)
print(generated_text)
