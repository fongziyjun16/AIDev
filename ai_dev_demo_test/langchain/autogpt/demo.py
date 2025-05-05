import os

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.tools import WriteFileTool, ReadFileTool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

os.environ["SERPAPI_API_KEY"] = "***"

search = SerpAPIWrapper()
tools = [
    Tool(name="search", func=search.run, description="useful for when you need to answer questions about current events. You should ask targeted questions"),
    WriteFileTool(),
    ReadFileTool(),
]

embeddings_model = OpenAIEmbeddings()

# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vector_store = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(model="gpt-4", temperature=0, verbose=True),
    memory=vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8} # 实例化 Faiss 的 VectorStoreRetriever
    ),
)

agent.chain.verbose = True

agent.run(["2022年冬奥会，中国金牌数是多少"])





