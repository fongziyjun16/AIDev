import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("讲个关于 {topic} 的笑话吧")
chain = prompt | model

def stream_demo():
    for s in chain.stream({"topic": "程序员"}):
        print(s.content, end="", flush=True)
# stream_demo()

def invoke_demo():
    print(chain.invoke({"topic": "程序员"}))
# invoke_demo()

def batch_demo():
    messages = chain.batch([{"topic": "程序员"}, {"topic": "产品经理"}, {"topic": "测试经理"}])
    output_parser = StrOutputParser()
    for idx, m in enumerate(messages):
        print(f"笑话{idx}:\n")
        print(output_parser.invoke(m))
        print("\n")
# batch_demo()

async def async_demo():
    async for s in chain.astream({"topic": "程序员"}):
        print(s.content, end="", flush=True)
# asyncio.run(async_demo())

async def await_demo():
    res = await chain.ainvoke({"topic": "程序员"})
    print(res.content)
asyncio.run(await_demo())


