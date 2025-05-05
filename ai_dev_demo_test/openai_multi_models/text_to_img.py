import os.path
import time

import requests

from utils.openai_client import get_openai_client

client = get_openai_client()

resp = client.images.generate(
    model="dall-e-3",
    prompt="a cat",
    size="1024x1024",
    quality="standard",
    n=1
)

def download_img(url, dest):
    resp = requests.get(url)
    if resp.status_code == 200:
        fn = str(time.time() * 1000) + ".jpeg"
        fp = os.path.join(dest, fn)
        with open(fn, "wb") as file:
            file.write(resp.content)
        print(f"Image successfully downloaded and saved to {fp}")
    else:
        print(f"Failed to download image. Status code: {resp.status_code}")

# print(resp.data[0].url)

download_img(resp.data[0].url, "./images")


