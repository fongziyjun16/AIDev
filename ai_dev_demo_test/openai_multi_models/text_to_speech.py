import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

speech_file_path = "./audio/tts_demo_01.mp3"

directory = os.path.dirname(speech_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

with openai.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="二营长！你他娘的意大利炮呢？给我拉来！"
) as resp:
    resp.stream_to_file(speech_file_path)

