from langchain.chains.llm import LLMChain
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

system_template = (
    """You are a translation expert, proficient in various languages. \n
    Translates {source_language} to {target_language}."""
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

m_chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt, human_message_prompt
])
m_translation_chain = LLMChain(llm=translation_model, prompt=m_chat_prompt_template)

res = m_translation_chain.run({
    "source_language": "Chinese",
    "target_language": "English",
    "text": "我喜欢学习大语言模型，轻松简单又愉快",
})
print(res)

res = m_translation_chain.run({
    "source_language": "Chinese",
    "target_language": "Japanese",
    "text": "我喜欢学习大语言模型，轻松简单又愉快",
})
print(res)


