import os

from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import OpenAI, ChatOpenAI

os.environ["SERPAPI_API_KEY"] = "***"

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

question = "谁是莱昂纳多·迪卡普里奥的女朋友？她现在年龄的0.43次方是多少?"

# 使用语言模型（gpt-3.5-turbo-instruct）
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke(question)

# 使用聊天模型（gpt-4o-mini）
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    tools, chat_model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
agent.invoke(question)