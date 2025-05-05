import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
models = openai.models.list()
model_ids = [model.id for model in models.data]
print(model_ids)
gpt_3 = openai.models.retrieve("gpt-3.5-turbo")
print(gpt_3)
