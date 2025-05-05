from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# 实例化查询工具
search = SerpAPIWrapper(
    serpapi_api_key="***"
)
tools = [
    Tool(
       name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

# 实例化 SELF_ASK_WITH_SEARCH Agent
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)

# 实际运行 Agent，查询问题（正确）
self_ask_with_search.invoke("成都举办的大运会是第几届大运会？2023年大运会举办地在哪里？")

# 实际运行 Agent，查询问题
self_ask_with_search.invoke("2023年大运会举办地在哪里？成都举办的大运会是第几届大运会？")

# Reason-only 正确：启发式 Prompt（猜测是大运会新闻报道数据给到了 gpt-3.5-turbo-instruct 模型）
print(llm.invoke("成都举办的大运会是第几届大运会？"))

# Reason-only 错误：非启发式 Prompt（容易出现事实类错误，未结合 Web Search 工具）
print(llm.invoke("2023年大运会举办地在哪里？"))

# 使用 GPT-4 作为大语言模型实现更优的 ReAct 范式
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
self_ask_with_search_chat = initialize_agent(
    tools, chat_model, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
# GPT-4 based ReAct 答案（正确）
self_ask_with_search_chat.invoke("成都举办的大运会是第几届大运会？2023年大运会举办地在哪里？")

