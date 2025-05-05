import os

import pandas as pd
import ast
import openai
import numpy as np

data_file = "data/AmazonFineFoodReviews_1k_embeddings.csv"
df = pd.read_csv(data_file)
df["embedding_vec"] = df["embedding"].apply(ast.literal_eval)
assert df["embedding_vec"].apply(len).nunique() == 1

openai.api_key = os.getenv("OPENAI_API_KEY")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
    resp = openai.embeddings.create(
        model=model,
        input=text
    )
    return resp.data[0].embedding

def search_review(data_frame, product_description, n=3, pprint=True):
    product_description_embedding = get_embedding(product_description)
    data_frame["similarity"] = data_frame.embedding_vec.apply(lambda x: cosine_similarity(x, product_description_embedding))
    results = (
        data_frame
            .sort_values("similarity", ascending=False)
            .head(n)
            .combined
                .str.replace("Title: ", "")
                .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

search_review(df, "delicious beans")