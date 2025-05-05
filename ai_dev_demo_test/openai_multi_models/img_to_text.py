import base64

import requests

from utils.openai_client import get_openai_client

openai_client = get_openai_client()

# resp = openai.chat.completions.create(
#     model="gpt-4-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "介绍下这幅图?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#                     },
#                 },
#             ],
#         }
#     ],
#     max_tokens=300
# )
#
# print(resp.choices[0].message.content)

def get_img_description(url, prompt="介绍下这幅图?"):
    resp = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        },
                    },
                ],
            }
        ],
        max_tokens=300
    )
    return resp.choices[0].message.content

# print(get_img_description(
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
# ))

def get_local_img_description(img_path, prompt="请解释图中内容？", max_tokens=1000):
    # base encode
    def encode_img(p):
        with open(p, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # get image base64
    img_base64 = encode_img(img_path)

    # construct http header
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_client.api_key}"
    }

    # construct request payload
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if resp.status_code == 200:
        resp_data = resp.json()
        content = resp_data['choices'][0]['message']['content']
        return content
    else:
        return f"Error: {resp.status_code}, {resp.text}"

print(get_local_img_description("./images/gdp_1980_2020.jpg"))