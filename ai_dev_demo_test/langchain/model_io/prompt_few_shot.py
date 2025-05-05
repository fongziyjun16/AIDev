from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

example_prompt = PromptTemplate(
    input_variables=["input", "output"],  # 输入变量的名字
    template="Input: {input}\nOutput: {output}",  # 实际的模板字符串
)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                          # 可供选择的示例列表
    OpenAIEmbeddings(),                # 用于生成嵌入向量的嵌入类，用于衡量语义相似性
    Chroma,                            # 用于存储嵌入向量并进行相似性搜索的 VectorStore 类
    k=1                                # 要生成的示例数量
)

simple_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 提供一个 ExampleSelector 替代示例
    example_prompt=example_prompt,      # 前面定义的提示模板
    prefix="Give the antonym of every input", # 前缀模板
    suffix="Input: {adjective}\nOutput:",     # 后缀模板
    input_variables=["adjective"],           # 输入变量的名字
)

print(simple_prompt.format(adjective="worried"))
print()

print(simple_prompt.format(adjective="long"))
print()

print(simple_prompt.format(adjective="rain"))
print()

