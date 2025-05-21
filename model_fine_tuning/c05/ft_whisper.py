from datasets import DatasetDict, load_dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq

model = "openai/whisper-large-v2"

dataset = "mozilla-foundation/common_voice_11_0"
language = "zh-CN"
task = "transcribe"

# data preparation
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset, language, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset, language, split="validation", trust_remote_code=True)

# model related
# 从预训练模型加载特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(model)
# 从预训练模型加载分词器，可以指定语言和任务以获得最适合特定需求的分词器配置
tokenizer = AutoTokenizer.from_pretrained(model, language=language, task=task)
# 从预训练模型加载处理器，处理器通常结合了特征提取器和分词器，为特定任务提供一站式的数据预处理
processor = AutoProcessor.from_pretrained(model, language=language, task=task)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model, load_in_8bits=True, device_map="auto")
