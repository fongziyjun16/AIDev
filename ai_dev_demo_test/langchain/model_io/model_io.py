from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAI, ChatOpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
# print(llm.invoke("Tell me a joke"))

chat_model = ChatOpenAI(model="gpt-3.5-turbo")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
    AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
    HumanMessage(content="Where was it played?")
]
resp = chat_model.invoke(messages)
print(resp.content)

