import pandas as pd
import tiktoken

data_file = "data/AmazonFineFoodReviews.csv"

df = pd.read_csv(data_file)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

top_n = 5

df = df.sort_values("Time").tail(top_n * 2)
df.drop("Time", axis=1, inplace=True)

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

max_tokens = 8000
df = df[df.n_tokens <= max_tokens].tail(top_n)

target_file = "data/AmazonFineFoodReviews_1k.csv"
df.to_csv(target_file)