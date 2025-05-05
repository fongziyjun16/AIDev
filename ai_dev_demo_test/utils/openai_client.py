import os

import openai


def get_openai_client():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai