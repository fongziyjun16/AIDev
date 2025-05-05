import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

text = "OpenAI provides powerful AI models for various applications."

response = openai.embeddings.create(
    model = embedding_model,
    input = text
)

print(response['data'][0]['embedding'])

