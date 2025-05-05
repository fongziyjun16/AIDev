from utils.openai_client import get_openai_client

openai_client = get_openai_client()

def get_text_from_speech(file_path):
    speech_file = open(file_path, "rb")
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=speech_file
    )
    return transcription.text

print(get_text_from_speech("./audios/liyunlong.mp3"))