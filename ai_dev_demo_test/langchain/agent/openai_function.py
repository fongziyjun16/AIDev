from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import tool, OpenAIFunctionsAgent, AgentExecutor

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

system_message = SystemMessage(content="你是非常强大的AI助手，但在计算单词长度方面不擅长。")
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)

agent = OpenAIFunctionsAgent(llm=chat_model, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke("单词“educa”中有多少个字母?")

MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

agent = OpenAIFunctionsAgent(llm=chat_model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
agent_executor.invoke("单词“educa”中有多少个字母?")
agent_executor.invoke("那是一个真实的单词吗？")



