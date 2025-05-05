import os

import pandas as pd
import openai

data_file = "data/AmazonFineFoodReviews_1k.csv"
target_file = "data/AmazonFineFoodReviews_1k_embeddings.csv"

df = pd.read_csv(data_file)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model):
    resp = openai.embeddings.create(
        model=model,
        input=text
    )
    return resp.data[0].embedding

embedding_model = "text-embedding-ada-002"
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, embedding_model))
df.to_csv(target_file)


